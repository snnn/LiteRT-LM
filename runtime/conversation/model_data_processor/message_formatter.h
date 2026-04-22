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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MESSAGE_FORMATTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MESSAGE_FORMATTER_H_

#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"

namespace litert::lm {

// MessageFormatter is the text-only subset of ModelDataProcessor needed for
// prompt rendering and response parsing.
class MessageFormatter {
 public:
  virtual ~MessageFormatter() = default;

  virtual absl::StatusOr<Message> TextToMessage(
      absl::string_view text,
      const DataProcessorArguments& args = DataProcessorArguments{}) const = 0;

  virtual absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const = 0;

  virtual absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const = 0;
};

template <typename ExpectedArgsT>
class TypeSafeMessageFormatter : public MessageFormatter {
 public:
  absl::StatusOr<Message> TextToMessage(
      absl::string_view text,
      const DataProcessorArguments& args = DataProcessorArguments{}) const final {
    if (std::holds_alternative<ExpectedArgsT>(args)) {
      return this->TextToMessageImpl(text, std::get<ExpectedArgsT>(args));
    }
    if (std::holds_alternative<std::monostate>(args)) {
      return this->TextToMessageImpl(text, ExpectedArgsT{});
    }
    return absl::InvalidArgumentError(
        "DataProcessorArguments does not hold the expected type");
  }

 private:
  virtual absl::StatusOr<Message> TextToMessageImpl(
      absl::string_view text, const ExpectedArgsT& typed_args) const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MESSAGE_FORMATTER_H_
