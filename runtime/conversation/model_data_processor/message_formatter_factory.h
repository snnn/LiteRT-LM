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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MESSAGE_FORMATTER_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MESSAGE_FORMATTER_FACTORY_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/message_formatter.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<MessageFormatter>> CreateMessageFormatter(
    const DataProcessorConfig& config,
    std::optional<Preface> preface = std::nullopt,
    PromptTemplateCapabilities capabilities = PromptTemplateCapabilities());

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MESSAGE_FORMATTER_FACTORY_H_
