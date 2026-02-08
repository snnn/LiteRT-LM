// Copyright 2025 The ODML Authors.
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

#include "runtime/conversation/conversation.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/internal_callback_util.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/model_type_utils.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

bool IsEmptyInputError(const absl::Status& status) {
  return absl::IsInvalidArgument(status) &&
         absl::StrContains(status.message(), "Input is empty");
}

// Ignores the invalid argument error when Session Prefill is called with empty
// input.
absl::Status IgnoreEmptyInputError(const absl::Status& status) {
  return IsEmptyInputError(status) ? absl::OkStatus() : status;
}
}  // namespace

absl::StatusOr<ConversationConfig> ConversationConfig::CreateDefault(
    const Engine& engine) {
  return ConversationConfig::Builder().Build(engine);
}

absl::StatusOr<ConversationConfig> ConversationConfig::CreateInternal(
    const Engine& engine, const SessionConfig& session_config,
    std::optional<Preface> preface,
    std::optional<PromptTemplate> overwrite_prompt_template,
    std::optional<DataProcessorConfig> overwrite_processor_config,
    bool enable_constrained_decoding, bool prefill_preface_on_init,
    std::optional<ConstraintProviderConfig> constraint_provider_config) {
  if (preface.has_value() && !std::holds_alternative<JsonPreface>(*preface)) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }

  SessionConfig session_config_copy = session_config;
  session_config_copy.SetApplyPromptTemplateInSession(false);
  RETURN_IF_ERROR(
      session_config_copy.MaybeUpdateAndValidate(engine.GetEngineSettings()));

  auto metadata = engine.GetEngineSettings().GetLlmMetadata();
  PromptTemplate prompt_template("");
  if (overwrite_prompt_template.has_value()) {
    prompt_template = *overwrite_prompt_template;
  } else if (metadata.has_value()) {
    if (metadata->has_jinja_prompt_template()) {
      prompt_template = PromptTemplate(metadata->jinja_prompt_template());
    } else if (metadata->has_prompt_templates()) {
      ASSIGN_OR_RETURN(
          std::string jinja_source,
          GetDefaultJinjaPromptTemplate(metadata->prompt_templates(),
                                        metadata->llm_model_type()));
      prompt_template = PromptTemplate(jinja_source);
    } else {
      return absl::InvalidArgumentError(
          "Failed to select jinja prompt template from llm metadata.");
    }
  } else {
    return absl::InvalidArgumentError(
        "Failed to select jinja prompt template. No llm metadata provided.");
  }

  DataProcessorConfig processor_config;
  if (overwrite_processor_config.has_value()) {
    // Use the overwrite processor config if provided.
    processor_config = *overwrite_processor_config;
  } else {
    // Build the processor config from the model metadata.
    ASSIGN_OR_RETURN(processor_config,
                     CreateDataProcessorConfigFromLlmModelType(
                         session_config_copy.GetLlmModelType()));
  }

  return ConversationConfig(
      session_config_copy, preface.value_or(JsonPreface()), prompt_template,
      processor_config, enable_constrained_decoding, prefill_preface_on_init,
      std::move(constraint_provider_config));
}

absl::StatusOr<std::string>
Conversation::GetSingleTurnTextFromSingleTurnTemplate(
    const JsonMessage& message, const OptionalArgs& optional_args) {
  absl::MutexLock lock(history_mutex_);  // NOLINT
  ASSIGN_OR_RETURN(
      auto result,
      model_data_processor_->RenderSingleTurnTemplate(
          history_,
          config_.prefill_preface_on_init() ? JsonPreface() : preface_, message,
          prompt_template_,
          /*current_is_appending_message=*/is_appending_message_,
          /*append_message=*/optional_args.has_pending_message));
  is_appending_message_ = result.is_appending_message;
  return result.text;
}

absl::StatusOr<std::string> Conversation::GetSingleTurnTextFromFullHistory(
    const JsonMessage& json_message, const OptionalArgs& optional_args) {
  PromptTemplateInput old_tmpl_input;
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_tmpl_input));

  absl::MutexLock lock(history_mutex_);  // NOLINT
  for (const auto& history_msg : history_) {
    if (std::holds_alternative<nlohmann::ordered_json>(history_msg)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(history_msg)));
      old_tmpl_input.messages.push_back(message_tmpl_input);
    } else {
      return absl::UnimplementedError("Message type is not supported yet");
    }
  }
  nlohmann::ordered_json messages =
      json_message.is_array() ? json_message
                              : nlohmann::ordered_json::array({json_message});
  if (history_.empty() && !config_.prefill_preface_on_init()) {
    PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
    for (const auto& message : messages) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(message));
      new_tmpl_input.messages.push_back(message_tmpl_input);
    }
    new_tmpl_input.add_generation_prompt = true;
    return prompt_template_.Apply(new_tmpl_input);
  }

  old_tmpl_input.add_generation_prompt = false;
  ASSIGN_OR_RETURN(const std::string old_string,
                   prompt_template_.Apply(old_tmpl_input));

  PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
  for (const auto& message : messages) {
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     model_data_processor_->MessageToTemplateInput(message));
    new_tmpl_input.messages.push_back(message_tmpl_input);
  }
  new_tmpl_input.add_generation_prompt = true;
  ASSIGN_OR_RETURN(const std::string& new_string,
                   prompt_template_.Apply(new_tmpl_input));
  if (new_string.substr(0, old_string.size()) != old_string) {
    return absl::InternalError(absl::StrCat(
        "The new rendered template string does not start with the previous "
        "rendered template string. \nold_string: ",
        old_string, "\nnew_string: ", new_string));
  }
  return {new_string.substr(old_string.size(),
                            new_string.size() - old_string.size())};
}

absl::StatusOr<std::string> Conversation::GetSingleTurnText(
    const Message& message, const OptionalArgs& optional_args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  nlohmann::ordered_json json_message =
      std::get<nlohmann::ordered_json>(message);
  if (!prompt_template_.GetCapabilities().supports_single_turn &&
      optional_args.has_pending_message) {
    return absl::InvalidArgumentError(
        "The prompt template does not support single turn template, but "
        "has_pending_message is true. `has_pending_message` is only valid for "
        "model templates and ModelDataProcessor that supports single turn "
        "prompt rendering.");
  }
  if (prompt_template_.GetCapabilities().supports_single_turn) {
    auto single_turn_text =
        GetSingleTurnTextFromSingleTurnTemplate(json_message, optional_args);
    if (!absl::IsUnimplemented(single_turn_text.status())) {
      return single_turn_text;
    }
  }
  return GetSingleTurnTextFromFullHistory(json_message, optional_args);
}

absl::StatusOr<DecodeConfig> Conversation::CreateDecodeConfig(
    std::optional<ConstraintArg> decoding_constraint) {
  auto decode_config = DecodeConfig::CreateDefault();
  if (decoding_constraint.has_value() && constraint_provider_ != nullptr) {
    ASSIGN_OR_RETURN(constraint_, constraint_provider_->CreateConstraint(
                                      std::move(decoding_constraint).value()));
  } else if (config_.constrained_decoding_enabled() && constraint_ == nullptr &&
             std::holds_alternative<JsonPreface>(preface_)) {
    // Create a constraint from the tools defined in the preface, if any.
    auto json_preface = std::get<JsonPreface>(preface_);
    if (!json_preface.tools.is_null()) {
      auto constraint =
          model_data_processor_->CreateConstraint(json_preface.tools);
      if (constraint.ok()) {
        constraint_ = std::move(constraint.value());
      } else if (!absl::IsUnimplemented(constraint.status())) {
        return constraint.status();
      }
    }
  }
  decode_config.SetConstraint(constraint_.get());
  return decode_config;
}

absl::StatusOr<std::unique_ptr<Conversation>> Conversation::Create(
    Engine& engine, const ConversationConfig& config) {
  absl::Time start_time = absl::Now();
  if (!std::holds_alternative<JsonPreface>(config.GetPreface())) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                   engine.CreateSession(config.GetSessionConfig()));
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config.GetProcessorConfig(), config.GetPreface(),
                               &session->GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config.constrained_decoding_enabled(),
                               config.GetPromptTemplate().GetCapabilities()));
  std::unique_ptr<ConstraintProvider> constraint_provider;
  if (config.constraint_provider_config().has_value()) {
    ASSIGN_OR_RETURN(constraint_provider,
                     CreateConstraintProvider(
                         config.constraint_provider_config().value(),
                         session->GetTokenizer(),
                         session->GetSessionConfig().GetStopTokenIds()));
  }
  auto conversation = absl::WrapUnique(new Conversation(
      std::move(session), std::move(model_data_processor), config.GetPreface(),
      config.GetPromptTemplate(), config, std::move(constraint_provider)));
  if (config.prefill_preface_on_init()) {
    std::string single_turn_text;
    std::vector<Message> tmp_history;
    const auto render_result =
        conversation->model_data_processor_->RenderSingleTurnTemplate(
            tmp_history, config.GetPreface(), JsonMessage(),
            config.GetPromptTemplate(),
            /*current_is_appending_message=*/false,
            /*append_message=*/false);
    if (absl::IsUnimplemented(render_result.status())) {
      // Fallback to the old way of prefilling the preface.
      PromptTemplateInput tmpl_input;
      RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
          config.GetPreface(), conversation->model_data_processor_.get(),
          tmpl_input));
      tmpl_input.add_generation_prompt = false;
      ASSIGN_OR_RETURN(single_turn_text,
                       conversation->prompt_template_.Apply(tmpl_input));
    } else if (render_result.ok()) {
      single_turn_text = render_result->text;
    } else {
      return render_result.status();
    }
    ASSIGN_OR_RETURN(const auto session_inputs,
                     conversation->model_data_processor_->ToInputDataVector(
                         single_turn_text,
                         std::get<JsonPreface>(config.GetPreface()).messages,
                         std::monostate()));
    if (!session_inputs.empty()) {
      RETURN_IF_ERROR(conversation->session_->RunPrefill(session_inputs));
    }
  }

  if (engine.GetEngineSettings().IsBenchmarkEnabled()) {
    ASSIGN_OR_RETURN(BenchmarkInfo * benchmark_info,
                     conversation->GetMutableBenchmarkInfo());
    RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kConversation, absl::Now() - start_time));
  }

  return conversation;
}

absl::StatusOr<Message> Conversation::SendMessage(const Message& message,
                                                  OptionalArgs optional_args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message, optional_args));
  absl::MutexLock lock(history_mutex_);  // NOLINT
  if (json_message.is_array()) {
    for (const auto& message : json_message) {
      history_.push_back(message);
    }
  } else {
    history_.push_back(json_message);
  }
  ASSIGN_OR_RETURN(
      const auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          optional_args.args.value_or(std::monostate())));
  RETURN_IF_ERROR(IgnoreEmptyInputError(session_->RunPrefill(session_inputs)));
  if (is_appending_message_) {
    return JsonMessage();
  } else {
    ASSIGN_OR_RETURN(
        auto decode_config,
        CreateDecodeConfig(std::move(optional_args.decoding_constraint)));
    ASSIGN_OR_RETURN(const Responses& responses,
                     session_->RunDecode(decode_config));
    ASSIGN_OR_RETURN(
        const Message assistant_message,
        model_data_processor_->ToMessage(
            responses, optional_args.args.value_or(std::monostate())));
    history_.push_back(assistant_message);
    return assistant_message;
  }
}

absl::Status Conversation::SendMessageAsync(
    const Message& message,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    OptionalArgs optional_args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message, optional_args));
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    if (json_message.is_array()) {
      for (const auto& message : json_message) {
        history_.push_back(message);
      }
    } else {
      history_.push_back(json_message);
    }
  }

  ASSIGN_OR_RETURN(
      const auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          optional_args.args.value_or(std::monostate())));

  absl::AnyInvocable<void(Message)> complete_message_callback =
      [this](const Message& complete_message) {
        absl::MutexLock lock(this->history_mutex_);  // NOLINT
        this->history_.push_back(complete_message);
      };

  absl::AnyInvocable<void()> cancel_callback = [this]() {
    absl::MutexLock lock(this->history_mutex_);  // NOLINT
    this->history_.pop_back();
  };

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> internal_callback =
      CreateInternalCallback(
          *model_data_processor_, optional_args.args.value_or(std::monostate()),
          std::move(user_callback), std::move(cancel_callback),
          std::move(complete_message_callback));

  ASSIGN_OR_RETURN(
      auto decode_config,
      CreateDecodeConfig(std::move(optional_args.decoding_constraint)));
  if (is_appending_message_) {
    ASSIGN_OR_RETURN(
        std::ignore,
        session_->RunPrefillAsync(
            session_inputs, [callback = std::move(internal_callback)](
                                absl::StatusOr<Responses> responses) mutable {
              auto status = IgnoreEmptyInputError(responses.status());
              if (!status.ok()) {
                callback(responses.status());
              }
            }));
  } else {
    ASSIGN_OR_RETURN(
        std::ignore,
        session_->RunPrefillAsync(
            session_inputs,
            [this, callback = std::move(internal_callback),
             decode_config](absl::StatusOr<Responses> responses) mutable {
              auto status = IgnoreEmptyInputError(responses.status());
              if (!status.ok()) {
                callback(responses.status());
              } else if (IsEmptyInputError(responses.status()) ||
                         responses->GetTaskState() == TaskState::kDone) {
                auto status = session_->RunDecodeAsync(std::move(callback),
                                                       decode_config);
              }
            }));
  }

  return absl::OkStatus();
};

absl::StatusOr<Responses> Conversation::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> cloned_session,
                   session_->Clone());
  return cloned_session->RunTextScoring(target_text,
                                        /*store_token_lengths=*/true);
}

absl::Status Conversation::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> cloned_session,
                   session_->CloneAsync(nullptr));
  return cloned_session
      ->RunTextScoringAsync(target_text, std::move(callback),
                            /*store_token_lengths=*/true)
      .status();
}

absl::StatusOr<BenchmarkInfo> Conversation::GetBenchmarkInfo() {
  return session_->GetBenchmarkInfo();
}

absl::StatusOr<BenchmarkInfo*> Conversation::GetMutableBenchmarkInfo() {
  return session_->GetMutableBenchmarkInfo();
}

void Conversation::CancelProcess() { session_->CancelProcess(); }

}  // namespace litert::lm
