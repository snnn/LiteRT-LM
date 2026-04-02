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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CHANNEL_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CHANNEL_UTIL_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Extracts channel content from responses and removes it from the responses
// in-place. Returns a map from channel name to extracted content.
absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ExtractChannelContent(const std::vector<Channel>& channels,
                      Responses& responses);

// Inserts extracted channel content into the assistant message under the
// "channels" key.
void InsertChannelContentIntoMessage(
    const absl::flat_hash_map<std::string, std::string>& channel_content,
    Message& assistant_message);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CHANNEL_UTIL_H_
