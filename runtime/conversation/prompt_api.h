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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_PROMPT_API_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_PROMPT_API_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"

namespace litert::lm {

struct PromptApiConfig {
  Preface preface = JsonPreface();
  PromptTemplate prompt_template = PromptTemplate("");
  DataProcessorConfig processor_config = GenericDataProcessorConfig();
  std::vector<Channel> channels;
};

struct PromptRenderOptions {
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  nlohmann::ordered_json tools = nullptr;
  nlohmann::ordered_json extra_context = nlohmann::ordered_json::object();
  bool add_generation_prompt = true;
  bool continue_final_message = false;
};

absl::StatusOr<PromptApiConfig> CreatePromptApiConfig(
    const proto::LlmMetadata& metadata, const proto::LlmModelType& model_type);

// Renders a prompt from structured messages using the model-native template
// and model-data-processor behavior. `continue_final_message` is intended for
// assistant-prefill cases such as a trailing "Answer:" stub in generation
// evals.
absl::StatusOr<std::string> RenderPrompt(const PromptApiConfig& config,
                                         const PromptRenderOptions& options);

// Parses raw generated text into a model-native assistant message using the
// model-data-processor and configured channels for the model.
absl::StatusOr<Message> ParseResponseText(const PromptApiConfig& config,
                                          absl::string_view text);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_PROMPT_API_H_
