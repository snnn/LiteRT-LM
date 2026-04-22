// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/prompt_api.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_processor_config_factory.h"
#include "runtime/conversation/model_data_processor/message_formatter.h"
#include "runtime/conversation/model_data_processor/message_formatter_factory.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/prompt_template_utils.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

absl::StatusOr<std::unique_ptr<MessageFormatter>> CreateDefaultFormatter(
    const PromptApiConfig& config) {
  return CreateMessageFormatter(config.processor_config, config.preface,
                                config.prompt_template.GetCapabilities());
}

absl::Status MergeExtraContext(const nlohmann::ordered_json& extra_context,
                              PromptTemplateInput& tmpl_input) {
  if (extra_context.is_null()) {
    return absl::OkStatus();
  }
  if (!extra_context.is_object()) {
    return absl::InvalidArgumentError("extra_context must be a JSON object.");
  }
  for (const auto& [key, value] : extra_context.items()) {
    tmpl_input.extra_context[key] = value;
  }
  return absl::OkStatus();
}

absl::Status AppendMessagesToTemplateInput(
    const nlohmann::ordered_json& messages,
    const MessageFormatter& message_formatter,
    PromptTemplateInput& tmpl_input) {
  if (!messages.is_array()) {
    return absl::InvalidArgumentError("messages must be a JSON array.");
  }
  for (const auto& message : messages) {
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     message_formatter.MessageToTemplateInput(message));
    tmpl_input.messages.push_back(std::move(message_tmpl_input));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> RenderPromptInternal(
    const PromptTemplate& prompt_template,
    const MessageFormatter& message_formatter, const Preface& preface,
    const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& extra_context, bool add_generation_prompt) {
  PromptTemplateInput tmpl_input;
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface, &message_formatter, tmpl_input));
  if (!tools.is_null()) {
    ASSIGN_OR_RETURN(tmpl_input.tools, message_formatter.FormatTools(tools));
  }
  RETURN_IF_ERROR(MergeExtraContext(extra_context, tmpl_input));
  RETURN_IF_ERROR(
      AppendMessagesToTemplateInput(messages, message_formatter, tmpl_input));
  tmpl_input.add_generation_prompt = add_generation_prompt;
  return prompt_template.Apply(tmpl_input);
}

absl::StatusOr<std::string> ContinueFinalAssistantMessage(
    const PromptTemplate& prompt_template,
    const MessageFormatter& message_formatter, const Preface& preface,
    const PromptRenderOptions& options) {
  if (!options.messages.is_array() || options.messages.empty()) {
    return absl::InvalidArgumentError(
        "continue_final_message requires a non-empty message array.");
  }
  nlohmann::ordered_json messages = options.messages;
  const auto& final_message = messages.back();
  if (!final_message.is_object() || !final_message.contains("role") ||
      final_message["role"] != "assistant" ||
      !final_message.contains("content") ||
      !final_message["content"].is_string()) {
    return absl::InvalidArgumentError(
        "continue_final_message requires the last message to be an assistant "
        "message with string content.");
  }
  std::string final_content = final_message["content"].get<std::string>();
  messages.erase(messages.end() - 1);
  ASSIGN_OR_RETURN(std::string prefix,
                   RenderPromptInternal(prompt_template, message_formatter,
                                        preface, messages, options.tools,
                                        options.extra_context,
                                        /*add_generation_prompt=*/true));
  return prefix + final_content;
}

absl::StatusOr<std::string> GetDefaultPromptTemplateSource(
    const proto::LlmMetadata& metadata) {
  if (metadata.has_jinja_prompt_template()) {
    return metadata.jinja_prompt_template();
  }
  if (metadata.has_prompt_templates()) {
    return GetDefaultJinjaPromptTemplate(metadata.prompt_templates(),
                                         metadata.llm_model_type());
  }
  return absl::InvalidArgumentError(
      "Failed to select jinja prompt template from llm metadata.");
}

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ExtractChannelContentFromText(
    const std::vector<Channel>& channels, std::string* content,
    const std::optional<std::string>& open_channel_name) {
  absl::flat_hash_map<std::string, std::string> extracted_fields;
  for (const auto& channel : channels) {
    std::string escaped_start = RE2::QuoteMeta(channel.start);
    std::string escaped_end = RE2::QuoteMeta(channel.end);
    RE2 re("(?s)(.*?)" + escaped_start + "(.*?)(" + escaped_end + "|$)");

    std::string channel_content;
    std::string new_content;
    absl::string_view remaining_content(*content);
    std::string text_before;
    std::string text_inside;
    std::string end_match;

    if (open_channel_name.has_value() &&
        *open_channel_name == channel.channel_name) {
      RE2 first_re("(?s)(.*?)(" + escaped_end + "|$)");
      if (RE2::Consume(&remaining_content, first_re, &text_inside,
                       &end_match)) {
        channel_content += text_inside;
      }
    }

    while (RE2::Consume(&remaining_content, re, &text_before, &text_inside,
                        &end_match)) {
      new_content += text_before;
      channel_content += text_inside;
    }
    new_content += std::string(remaining_content);

    if (!channel_content.empty()) {
      *content = std::move(new_content);
      extracted_fields[channel.channel_name] += channel_content;
    }
  }
  return extracted_fields;
}

void InsertChannelContentIntoMessage(
    const absl::flat_hash_map<std::string, std::string>& channel_content,
    Message& assistant_message) {
  for (const auto& [channel_name, value] : channel_content) {
    assistant_message["channels"][channel_name] = value;
  }
}

std::optional<std::string> GetOpenChannelName(
    absl::string_view text, const std::vector<Channel>& channels) {
  std::optional<std::string> open_channel_name;
  size_t max_start_pos = std::string::npos;

  for (const auto& channel : channels) {
    if (channel.start.empty()) {
      continue;
    }

    size_t last_start = text.rfind(channel.start);
    if (last_start == absl::string_view::npos) {
      continue;
    }

    size_t last_end =
        channel.end.empty() ? absl::string_view::npos : text.rfind(channel.end);

    bool is_open =
        (last_end == absl::string_view::npos) || (last_start > last_end);
    if (is_open &&
        (max_start_pos == std::string::npos || last_start > max_start_pos)) {
      max_start_pos = last_start;
      open_channel_name = channel.channel_name;
    }
  }

  return open_channel_name;
}

}  // namespace

absl::StatusOr<PromptApiConfig> CreatePromptApiConfig(
    const proto::LlmMetadata& metadata, const proto::LlmModelType& model_type) {
  ASSIGN_OR_RETURN(std::string jinja_source,
                   GetDefaultPromptTemplateSource(metadata));
  ASSIGN_OR_RETURN(DataProcessorConfig processor_config,
                   CreateDataProcessorConfigFromLlmModelType(model_type));

  PromptApiConfig config;
  config.prompt_template = PromptTemplate(jinja_source);
  config.processor_config = std::move(processor_config);
  for (const auto& channel : metadata.channels()) {
    config.channels.push_back(
        litert::lm::Channel{.channel_name = channel.channel_name(),
                            .start = channel.start(),
                            .end = channel.end()});
  }
  for (const auto& channel : config.channels) {
    if (channel.channel_name.empty()) {
      return absl::InvalidArgumentError(
          "Custom channel must have a non-empty channel_name.");
    }
  }
  return config;
}

absl::StatusOr<std::string> RenderPrompt(const PromptApiConfig& config,
                                         const PromptRenderOptions& options) {
  ASSIGN_OR_RETURN(std::unique_ptr<MessageFormatter> message_formatter,
                   CreateDefaultFormatter(config));
  if (options.continue_final_message) {
    return ContinueFinalAssistantMessage(config.prompt_template,
                                         *message_formatter, config.preface,
                                         options);
  }
  return RenderPromptInternal(config.prompt_template, *message_formatter,
                              config.preface, options.messages, options.tools,
                              options.extra_context,
                              options.add_generation_prompt);
}

absl::StatusOr<Message> ParseResponseText(const PromptApiConfig& config,
                                          absl::string_view text) {
  ASSIGN_OR_RETURN(std::unique_ptr<MessageFormatter> message_formatter,
                   CreateDefaultFormatter(config));
  std::vector<Channel> custom_channels;
  for (const auto& channel : config.channels) {
    if (!channel.channel_name.empty()) {
      custom_channels.push_back(channel);
    }
  }
  std::string response_text(text);
  auto open_channel_name = GetOpenChannelName(text, custom_channels);
  ASSIGN_OR_RETURN(
      auto extracted_channels,
      ExtractChannelContentFromText(custom_channels, &response_text,
                                    open_channel_name));
  ASSIGN_OR_RETURN(
      Message message,
      message_formatter->TextToMessage(response_text, DataProcessorArguments{}));
  InsertChannelContentIntoMessage(extracted_channels, message);
  return message;
}

}  // namespace litert::lm
