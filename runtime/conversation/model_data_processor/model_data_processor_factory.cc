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

#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> CreateModelDataProcessor(
    const DataProcessorConfig& config, std::optional<Preface> preface,
    const Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    bool enable_constrained_decoding, PromptTemplateCapabilities capabilities) {
  if (std::holds_alternative<Gemma3DataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating Gemma3DataProcessor";
    return Gemma3DataProcessor::Create(
        std::get<Gemma3DataProcessorConfig>(config), preface, tokenizer,
        stop_token_ids, enable_constrained_decoding);
  } else if (std::holds_alternative<Qwen3DataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating Qwen3DataProcessor";
    return Qwen3DataProcessor::Create(
        std::get<Qwen3DataProcessorConfig>(config), preface);
  } else if (std::holds_alternative<GenericDataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating GenericDataProcessor";
    return GenericDataProcessor::Create(
        std::get<GenericDataProcessorConfig>(config), capabilities);
  } else if (std::holds_alternative<FunctionGemmaDataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating FunctionGemmaDataProcessor";
    return FunctionGemmaDataProcessor::Create(
        std::get<FunctionGemmaDataProcessorConfig>(config), preface, tokenizer,
        stop_token_ids, enable_constrained_decoding);
  } else if (std::holds_alternative<Gemma4DataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating Gemma4DataProcessor";
    return Gemma4DataProcessor::Create(
        std::get<Gemma4DataProcessorConfig>(config), preface, tokenizer,
        stop_token_ids, enable_constrained_decoding);
  } else {
    return absl::InvalidArgumentError("Unsupported data processor config type");
  }
}

}  // namespace litert::lm
