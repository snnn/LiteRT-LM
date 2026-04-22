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

#include "runtime/conversation/model_data_processor/message_formatter_factory.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/fc_tool_format_utils.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/components/tool_use/python_tool_format_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/message_formatter.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

bool HasToolPreface(const std::optional<Preface>& preface) {
  return preface.has_value() && std::holds_alternative<JsonPreface>(*preface) &&
         !std::get<JsonPreface>(*preface).tools.empty();
}

bool HasToolCalls(const ordered_json& message) {
  return message.contains("tool_calls") && message["tool_calls"].is_array();
}

bool IsToolRoleMessage(const ordered_json& message) {
  return message.contains("role") && message["role"] == "tool";
}

absl::StatusOr<std::string> FormatPythonToolResponse(
    const ordered_json& tool_response) {
  absl::string_view tool_response_key;
  if (tool_response.contains("tool_response")) {
    tool_response_key = "tool_response";
  } else if (tool_response.contains("response")) {
    tool_response_key = "response";
  } else {
    return FormatValueAsPython(tool_response);
  }
  return FormatValueAsPython(tool_response[tool_response_key]);
}

absl::StatusOr<std::string> FormatFcToolResponse(
    const ordered_json& tool_response, absl::string_view escape_tag) {
  std::optional<std::string> tool_name;
  if (tool_response.contains("name") && tool_response["name"].is_string()) {
    tool_name = tool_response["name"].get<std::string>();
  } else if (tool_response.contains("tool_name") &&
             tool_response["tool_name"].is_string()) {
    tool_name = tool_response["tool_name"].get<std::string>();
  }

  if (!tool_name.has_value()) {
    return FormatValueAsFc(tool_response, escape_tag);
  }

  ordered_json response;
  if (tool_response.contains("response") &&
      tool_response["response"].is_object()) {
    response = tool_response["response"];
  } else if (tool_response.contains("value") &&
             tool_response["value"].is_object()) {
    response = tool_response["value"];
  }

  if (!response.is_null()) {
    ASSIGN_OR_RETURN(std::string value, FormatValueAsFc(response, escape_tag));
    return absl::StrCat(*tool_name, value);
  }

  ordered_json fields = tool_response;
  fields.erase("tool_name");
  fields.erase("name");
  ASSIGN_OR_RETURN(std::string value, FormatValueAsFc(fields, escape_tag));
  return absl::StrCat(*tool_name, value);
}

absl::StatusOr<ordered_json> FormatFcToolResponses(const ordered_json& content,
                                                   absl::string_view escape_tag) {
  if (content.is_object()) {
    return FormatFcToolResponse(content, escape_tag);
  }
  if (content.is_array()) {
    ordered_json tool_content = ordered_json::array();
    for (const auto& item : content) {
      ordered_json tool_response =
          item.contains("tool_response") ? item["tool_response"] : item;
      ASSIGN_OR_RETURN(std::string formatted_tool_response,
                       FormatFcToolResponse(tool_response, escape_tag));
      tool_content.push_back(
          {{"type", "text"}, {"text", formatted_tool_response}});
    }
    return tool_content;
  }
  return content;
}

class GenericMessageFormatter
    : public TypeSafeMessageFormatter<GenericDataProcessorArguments> {
 public:
  GenericMessageFormatter(GenericDataProcessorConfig config,
                          PromptTemplateCapabilities capabilities)
      : config_(std::move(config)), capabilities_(capabilities) {}

  absl::StatusOr<ordered_json> MessageToTemplateInput(
      const ordered_json& message) const override {
    if (message["content"].is_string() && capabilities_.requires_typed_content) {
      return ordered_json::object(
          {{"role", message["role"]},
           {"content",
            ordered_json::array({{{"type", "text"}, {"text", message["content"]}}})}});
    }
    if (message["content"].is_array() && message["content"].size() == 1 &&
        message["content"][0]["type"] == "text" &&
        !capabilities_.requires_typed_content) {
      return ordered_json::object(
          {{"role", message["role"]}, {"content", message["content"][0]["text"]}});
    }
    return message;
  }

  absl::StatusOr<ordered_json> FormatTools(
      const ordered_json& tools) const override {
    return tools;
  }

 private:
  absl::StatusOr<Message> TextToMessageImpl(
      absl::string_view response_text,
      const GenericDataProcessorArguments& args) const override {
    ordered_json content;
    if (config_.force_string_content) {
      content = response_text;
    } else {
      content = ordered_json::array(
          {{{"type", "text"}, {"text", std::string(response_text)}}});
    }
    return ordered_json::object(
        {{"role", config_.model_role}, {"content", content}});
  }

  GenericDataProcessorConfig config_;
  PromptTemplateCapabilities capabilities_;
};

class Qwen3MessageFormatter
    : public TypeSafeMessageFormatter<Qwen3DataProcessorArguments> {
 public:
  Qwen3MessageFormatter(Qwen3DataProcessorConfig config,
                        std::optional<Preface> preface)
      : config_(std::move(config)), preface_(std::move(preface)) {}

  absl::StatusOr<ordered_json> MessageToTemplateInput(
      const ordered_json& message) const override {
    if (message["content"].is_array()) {
      const auto& content = message["content"];
      if (content.size() == 1 && content[0].contains("text")) {
        return ordered_json::object(
            {{"role", message["role"]}, {"content", content[0]["text"]}});
      }
    }
    return message;
  }

  absl::StatusOr<ordered_json> FormatTools(
      const ordered_json& tools) const override {
    return tools;
  }

 private:
  absl::StatusOr<Message> TextToMessageImpl(
      absl::string_view response_text,
      const Qwen3DataProcessorArguments& args) const override {
    ordered_json message = {{"role", "assistant"}};
    if (HasToolPreface(preface_)) {
      ASSIGN_OR_RETURN(ordered_json content_and_tool_calls,
                       ParseTextAndToolCalls(
                           response_text, config_.code_fence_start,
                           config_.code_fence_end, SyntaxType::kJson,
                           config_.escape_fence_strings,
                           config_.tool_code_regex));
      if (content_and_tool_calls.contains("content")) {
        message["content"] = content_and_tool_calls["content"];
      }
      if (content_and_tool_calls.contains("tool_calls")) {
        message["tool_calls"] = content_and_tool_calls["tool_calls"];
      }
    } else {
      message["content"] = ordered_json::array(
          {{{"type", "text"}, {"text", std::string(response_text)}}});
    }
    return message;
  }

  Qwen3DataProcessorConfig config_;
  std::optional<Preface> preface_;
};

class FunctionGemmaMessageFormatter
    : public TypeSafeMessageFormatter<FunctionGemmaDataProcessorArguments> {
 public:
  FunctionGemmaMessageFormatter(FunctionGemmaDataProcessorConfig config,
                                std::optional<Preface> preface)
      : config_(std::move(config)), preface_(std::move(preface)) {}

  absl::StatusOr<ordered_json> MessageToTemplateInput(
      const ordered_json& message) const override {
    if (config_.use_template_for_fc_format) {
      return message;
    }
    if (!message.contains("tool_calls") && message["role"] != "tool") {
      return message;
    }

    ordered_json template_input = ordered_json::object();
    if (message.contains("role")) {
      template_input["role"] = message["role"];
    }
    if (message.contains("content")) {
      if (template_input.contains("role") && template_input["role"] == "tool") {
        ASSIGN_OR_RETURN(template_input["content"],
                         FormatFcToolResponses(message["content"], "<escape>"));
      } else {
        template_input["content"] = message["content"];
      }
    }
    if (message.contains("tool_calls")) {
      template_input["tool_calls"] = ordered_json::array();
      for (const auto& tool_call : message["tool_calls"]) {
        if (!tool_call.contains("function")) {
          continue;
        }
        const ordered_json& function = tool_call["function"];
        ordered_json tool_call_input = ordered_json::object();
        tool_call_input["type"] = "function";
        tool_call_input["function"]["name"] = function["name"];
        if (function.contains("arguments")) {
          if (function["arguments"].is_object()) {
            for (const auto& [key, value] : function["arguments"].items()) {
              ASSIGN_OR_RETURN(std::string formatted_value,
                               FormatValueAsFc(value));
              tool_call_input["function"]["arguments"][key] = formatted_value;
            }
          } else {
            tool_call_input["function"]["arguments"] = function["arguments"];
          }
        }
        template_input["tool_calls"].push_back(tool_call_input);
      }
    }
    return template_input;
  }

  absl::StatusOr<ordered_json> FormatTools(
      const ordered_json& tools) const override {
    if (config_.use_template_for_fc_format) {
      return tools;
    }
    if (!tools.is_array()) {
      return absl::InvalidArgumentError("Tools must be an array.");
    }
    ordered_json formatted_tools = ordered_json::array();
    for (const auto& tool : tools) {
      ASSIGN_OR_RETURN(std::string formatted_tool, FormatToolAsFc(tool));
      formatted_tools.push_back(formatted_tool);
    }
    return formatted_tools;
  }

 private:
  absl::StatusOr<Message> TextToMessageImpl(
      absl::string_view response_text,
      const FunctionGemmaDataProcessorArguments& args) const override {
    ordered_json message = {{"role", "assistant"}};
    if (HasToolPreface(preface_)) {
      ASSIGN_OR_RETURN(ordered_json content_and_tool_calls,
                       ParseTextAndToolCalls(
                           response_text, config_.code_fence_start,
                           config_.code_fence_end,
                           GetSyntaxType(config_.syntax_type),
                           config_.escape_fence_strings,
                           config_.tool_code_regex));
      if (content_and_tool_calls.contains("content")) {
        message["content"] = content_and_tool_calls["content"];
      }
      if (content_and_tool_calls.contains("tool_calls")) {
        message["tool_calls"] = content_and_tool_calls["tool_calls"];
      }
    } else {
      message["content"] = ordered_json::array(
          {{{"type", "text"}, {"text", std::string(response_text)}}});
    }
    return message;
  }

  FunctionGemmaDataProcessorConfig config_;
  std::optional<Preface> preface_;
};

class Gemma3MessageFormatter
    : public TypeSafeMessageFormatter<Gemma3DataProcessorArguments> {
 public:
  Gemma3MessageFormatter(Gemma3DataProcessorConfig config,
                         std::optional<Preface> preface)
      : config_(std::move(config)), preface_(std::move(preface)) {}

  absl::StatusOr<ordered_json> MessageToTemplateInput(
      const ordered_json& message) const override {
    if (!HasToolCalls(message) && !IsToolRoleMessage(message)) {
      return message;
    }

    ordered_json template_input = ordered_json::object();
    if (message.contains("role")) {
      template_input["role"] = message["role"];
    }
    if (message.contains("content")) {
      if (IsToolRoleMessage(message)) {
        if (message["content"].is_array()) {
          template_input["content"] = ordered_json::array();
          for (const auto& item : message["content"]) {
            ASSIGN_OR_RETURN(std::string formatted_tool_response,
                             FormatPythonToolResponse(item));
            template_input["content"].push_back(
                {{"type", "text"}, {"text", formatted_tool_response}});
          }
        } else if (message["content"].is_object()) {
          ASSIGN_OR_RETURN(std::string formatted_tool_response,
                           FormatPythonToolResponse(message["content"]));
          template_input["content"] = formatted_tool_response;
        } else {
          template_input["content"] = message["content"];
        }
      } else {
        template_input["content"] = message["content"];
      }
    }
    if (message.contains("tool_calls")) {
      template_input["tool_calls"] = ordered_json::array();
      for (const auto& tool_call : message["tool_calls"]) {
        if (!tool_call.contains("function")) {
          continue;
        }
        const ordered_json& function = tool_call["function"];
        ordered_json tool_call_input = ordered_json::object();
        tool_call_input["type"] = "function";
        tool_call_input["function"]["name"] = function["name"];
        if (function.contains("arguments")) {
          if (function["arguments"].is_object()) {
            for (const auto& [key, value] : function["arguments"].items()) {
              ASSIGN_OR_RETURN(std::string formatted_value,
                               FormatValueAsPython(value));
              tool_call_input["function"]["arguments"][key] = formatted_value;
            }
          } else {
            tool_call_input["function"]["arguments"] = function["arguments"];
          }
        }
        template_input["tool_calls"].push_back(tool_call_input);
      }
    }
    return template_input;
  }

  absl::StatusOr<ordered_json> FormatTools(
      const ordered_json& tools) const override {
    if (!tools.is_array()) {
      return absl::InvalidArgumentError("Tools must be an array.");
    }
    ordered_json formatted_tools = ordered_json::array();
    for (const auto& tool : tools) {
      ASSIGN_OR_RETURN(std::string formatted_tool, FormatToolAsPython(tool));
      formatted_tools.push_back(formatted_tool);
    }
    return formatted_tools;
  }

 private:
  absl::StatusOr<Message> TextToMessageImpl(
      absl::string_view response_text,
      const Gemma3DataProcessorArguments& args) const override {
    ordered_json message = {{"role", "assistant"}};
    if (HasToolPreface(preface_)) {
      ASSIGN_OR_RETURN(ordered_json content_and_tool_calls,
                       ParseTextAndToolCalls(
                           response_text, config_.code_fence_start,
                           config_.code_fence_end,
                           GetSyntaxType(config_.syntax_type),
                           config_.escape_fence_strings,
                           config_.tool_code_regex));
      if (content_and_tool_calls.contains("content")) {
        message["content"] = content_and_tool_calls["content"];
      }
      if (content_and_tool_calls.contains("tool_calls")) {
        message["tool_calls"] = content_and_tool_calls["tool_calls"];
      }
    } else {
      message["content"] = ordered_json::array(
          {{{"type", "text"}, {"text", std::string(response_text)}}});
    }
    return message;
  }

  Gemma3DataProcessorConfig config_;
  std::optional<Preface> preface_;
};

class Gemma4MessageFormatter
    : public TypeSafeMessageFormatter<Gemma4DataProcessorArguments> {
 public:
  Gemma4MessageFormatter(Gemma4DataProcessorConfig config,
                         std::optional<Preface> preface)
      : config_(std::move(config)), preface_(std::move(preface)) {}

  absl::StatusOr<ordered_json> MessageToTemplateInput(
      const ordered_json& message) const override {
    if (config_.use_template_for_fc_format) {
      return message;
    }
    if (!message.contains("tool_calls") && message["role"] != "tool") {
      return message;
    }

    ordered_json template_input = ordered_json::object();
    if (message.contains("role")) {
      template_input["role"] = message["role"];
    }
    if (message.contains("content")) {
      if (template_input.contains("role") && template_input["role"] == "tool") {
        ASSIGN_OR_RETURN(template_input["content"],
                         FormatFcToolResponses(message["content"],
                                               config_.open_quote));
      } else {
        template_input["content"] = message["content"];
      }
    }
    if (message.contains("tool_calls")) {
      template_input["tool_calls"] = ordered_json::array();
      for (const auto& tool_call : message["tool_calls"]) {
        if (!tool_call.contains("function")) {
          continue;
        }
        const ordered_json& function = tool_call["function"];
        ordered_json tool_call_input = ordered_json::object();
        tool_call_input["type"] = "function";
        tool_call_input["function"]["name"] = function["name"];
        if (function.contains("arguments")) {
          if (function["arguments"].is_object()) {
            for (const auto& [key, value] : function["arguments"].items()) {
              ASSIGN_OR_RETURN(std::string formatted_value,
                               FormatValueAsFc(value, config_.open_quote));
              tool_call_input["function"]["arguments"][key] = formatted_value;
            }
          } else {
            tool_call_input["function"]["arguments"] = function["arguments"];
          }
        }
        template_input["tool_calls"].push_back(tool_call_input);
      }
    }
    return template_input;
  }

  absl::StatusOr<ordered_json> FormatTools(
      const ordered_json& tools) const override {
    if (config_.use_template_for_fc_format) {
      return tools;
    }
    if (!tools.is_array()) {
      return absl::InvalidArgumentError("Tools must be an array.");
    }
    ordered_json formatted_tools = ordered_json::array();
    for (const auto& tool : tools) {
      ASSIGN_OR_RETURN(std::string formatted_tool,
                       FormatToolAsFc(tool, config_.open_quote));
      formatted_tools.push_back(formatted_tool);
    }
    return formatted_tools;
  }

 private:
  absl::StatusOr<Message> TextToMessageImpl(
      absl::string_view response_text,
      const Gemma4DataProcessorArguments& args) const override {
    ordered_json message = {{"role", "assistant"}};
    if (HasToolPreface(preface_)) {
      ASSIGN_OR_RETURN(ordered_json content_and_tool_calls,
                       ParseTextAndToolCalls(
                           response_text, config_.code_fence_start,
                           config_.code_fence_end,
                           GetSyntaxType(config_.syntax_type),
                           config_.escape_fence_strings,
                           config_.tool_code_regex));
      if (content_and_tool_calls.contains("content")) {
        message["content"] = content_and_tool_calls["content"];
      }
      if (content_and_tool_calls.contains("tool_calls")) {
        message["tool_calls"] = content_and_tool_calls["tool_calls"];
      }
    } else {
      message["content"] = ordered_json::array(
          {{{"type", "text"}, {"text", std::string(response_text)}}});
    }
    return message;
  }

  Gemma4DataProcessorConfig config_;
  std::optional<Preface> preface_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<MessageFormatter>> CreateMessageFormatter(
    const DataProcessorConfig& config, std::optional<Preface> preface,
    PromptTemplateCapabilities capabilities) {
  if (std::holds_alternative<GenericDataProcessorConfig>(config)) {
    std::unique_ptr<MessageFormatter> formatter =
        std::make_unique<GenericMessageFormatter>(
            std::get<GenericDataProcessorConfig>(config), capabilities);
    return formatter;
  }
  if (std::holds_alternative<Qwen3DataProcessorConfig>(config)) {
    std::unique_ptr<MessageFormatter> formatter =
        std::make_unique<Qwen3MessageFormatter>(
            std::get<Qwen3DataProcessorConfig>(config), std::move(preface));
    return formatter;
  }
  if (std::holds_alternative<FunctionGemmaDataProcessorConfig>(config)) {
    std::unique_ptr<MessageFormatter> formatter =
        std::make_unique<FunctionGemmaMessageFormatter>(
            std::get<FunctionGemmaDataProcessorConfig>(config),
            std::move(preface));
    return formatter;
  }
  if (std::holds_alternative<Gemma3DataProcessorConfig>(config)) {
    std::unique_ptr<MessageFormatter> formatter =
        std::make_unique<Gemma3MessageFormatter>(
            std::get<Gemma3DataProcessorConfig>(config), std::move(preface));
    return formatter;
  }
  if (std::holds_alternative<Gemma4DataProcessorConfig>(config)) {
    std::unique_ptr<MessageFormatter> formatter =
        std::make_unique<Gemma4MessageFormatter>(
            std::get<Gemma4DataProcessorConfig>(config), std::move(preface));
    return formatter;
  }
  return absl::InvalidArgumentError("Unsupported data processor config.");
}

}  // namespace litert::lm
