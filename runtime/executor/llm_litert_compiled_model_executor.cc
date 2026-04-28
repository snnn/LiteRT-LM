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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_profiler.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/common_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"
#include "runtime/executor/llm_litert_mtp_drafter.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/log_tensor_buffer.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert
#include "tflite/types/half.h"  // from @litert

#if defined(__linux__) && defined(__has_include)
#if __has_include(<sys/sdt.h>)
#include <sys/sdt.h>
#define LITERT_LM_USDT_PHASE_BEGIN(step, layer, phase) \
  STAP_PROBE3(litert_lm_exec, phase_begin, step, layer, phase)
#define LITERT_LM_USDT_PHASE_END(step, layer, phase) \
  STAP_PROBE3(litert_lm_exec, phase_end, step, layer, phase)
#else
#define LITERT_LM_USDT_PHASE_BEGIN(step, layer, phase)
#define LITERT_LM_USDT_PHASE_END(step, layer, phase)
#endif
#else
#define LITERT_LM_USDT_PHASE_BEGIN(step, layer, phase)
#define LITERT_LM_USDT_PHASE_END(step, layer, phase)
#endif

namespace litert::lm {
namespace {

bool ShouldProfileScoring() {
  static const bool enabled = []() {
    const char* value = std::getenv("LITERT_LM_PROFILE_SCORING");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

bool ShouldProfileKvCopyLocks() {
  static const bool enabled = []() {
    const char* value = std::getenv("LITERT_LM_PROFILE_KV_COPY_LOCKS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

bool ShouldDebugInputPosLock() {
  static const bool enabled = []() {
    const char* value = std::getenv("LITERT_LM_DEBUG_INPUT_POS_LOCK");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

std::string GetFirstDecodeDumpDirFromEnv() {
  const char* value = std::getenv("LITERT_LM_DUMP_FIRST_DECODE_TENSORS_DIR");
  return value == nullptr ? std::string() : std::string(value);
}

std::string GetFirstDecodeProfilePathFromEnv() {
  const char* value = std::getenv("LITERT_LM_DUMP_FIRST_DECODE_PROFILE_PATH");
  return value == nullptr ? std::string() : std::string(value);
}

bool ShouldEnableFirstDecodeProfiling(
    const AdvancedSettings& advanced_settings) {
  return !advanced_settings.dump_first_decode_profile_path.empty() ||
         !GetFirstDecodeProfilePathFromEnv().empty();
}

using ::absl::Span;

inline void BenchmarkMark(BenchmarkInfo* benchmark_info,
                          absl::string_view mark_name) {
  if (benchmark_info != nullptr) {
    benchmark_info->TimeMarkDelta(std::string(mark_name)).IgnoreError();
  }
}

enum ExecutorTracePhase : int32_t {
  kTraceUnknown = 0,
  kTraceDecodePrepareInputs = 1,
  kTraceDecodeEmbeddingLookup = 2,
  kTraceDecodeBindBuffers = 3,
  kTraceDecodeModelRun = 4,
  kTraceDecodeSampling = 5,
  kTracePrefillPrepareInputs = 6,
  kTracePrefillEmbeddingLookup = 7,
  kTracePrefillBindBuffers = 8,
  kTracePrefillModelRun = 9,
  kTracePrepareFirstDecode = 10,
  kTracePrepareFirstDecodeBroadcast = 11,
  kTracePrepareFirstDecodeCopyKv = 12,
};

inline void TracePhaseBegin(int step, ExecutorTracePhase phase, int layer = -1) {
  LITERT_LM_USDT_PHASE_BEGIN(step, layer, static_cast<int32_t>(phase));
}

inline void TracePhaseEnd(int step, ExecutorTracePhase phase, int layer = -1) {
  LITERT_LM_USDT_PHASE_END(step, layer, static_cast<int32_t>(phase));
}

std::string SanitizeTensorName(absl::string_view name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}

std::string ElementTypeName(litert::ElementType element_type) {
  switch (element_type) {
    case litert::ElementType::Float32:
      return "FLOAT32";
    case litert::ElementType::Float16:
      return "FLOAT16";
    case litert::ElementType::Int32:
      return "INT32";
    case litert::ElementType::Int8:
      return "INT8";
    case litert::ElementType::Bool:
      return "BOOL";
    default:
      return absl::StrCat(static_cast<int>(element_type));
  }
}

std::string ShapeString(const litert::RankedTensorType& tensor_type) {
  const auto dims = tensor_type.Layout().Dimensions();
  if (dims.empty()) {
    return "scalar";
  }
  std::string out;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      out.push_back('x');
    }
    absl::StrAppend(&out, dims[i]);
  }
  return out;
}

std::string StridesString(const litert::RankedTensorType& tensor_type);

std::string BufferTypeName(const TensorBuffer& tensor) {
  auto buffer_type = tensor.BufferType();
  if (!buffer_type.HasValue()) {
    return absl::StrCat("ERROR(", buffer_type.Error().Message(), ")");
  }
  return BufferTypeToStringCC(*buffer_type);
}

std::string TensorBufferSummary(const TensorBuffer& tensor) {
  std::vector<std::string> parts;
  parts.push_back(absl::StrCat("buffer_type=", BufferTypeName(tensor)));
  if (auto tensor_type = tensor.TensorType(); tensor_type.HasValue()) {
    parts.push_back(
        absl::StrCat("elem_type=", ElementTypeName(tensor_type->ElementType())));
    parts.push_back(absl::StrCat("shape=", ShapeString(*tensor_type)));
    const std::string strides = StridesString(*tensor_type);
    if (!strides.empty()) {
      parts.push_back(absl::StrCat("strides=", strides));
    }
  } else {
    parts.push_back(
        absl::StrCat("tensor_type_error=", tensor_type.Error().Message()));
  }
  if (auto packed_size = tensor.PackedSize(); packed_size.HasValue()) {
    parts.push_back(absl::StrCat("packed_size=", *packed_size));
  } else {
    parts.push_back(
        absl::StrCat("packed_size_error=", packed_size.Error().Message()));
  }
  return absl::StrJoin(parts, " ");
}

bool ShouldTraceKvTensorName(absl::string_view name) {
  return name.find("kv_cache_k_13") != absl::string_view::npos ||
         name.find("kv_cache_v_13") != absl::string_view::npos ||
         name.find("kv_cache_k_14") != absl::string_view::npos ||
         name.find("kv_cache_v_14") != absl::string_view::npos;
}

void MaybeLogCompiledModelBufferCreation(
    absl::string_view phase, absl::string_view signature_name,
    absl::string_view tensor_name, bool is_input,
    const TensorBuffer& tensor_buffer) {
  if (!ShouldTraceKvTensorName(tensor_name)) {
    return;
  }
  ABSL_LOG(INFO) << "Executor buffer create phase=" << phase
                 << " signature=" << signature_name
                 << " direction=" << (is_input ? "input" : "output")
                 << " name=" << tensor_name
                 << " created_type=" << BufferTypeName(tensor_buffer);
}

std::string StridesString(const litert::RankedTensorType& tensor_type) {
  const auto layout = tensor_type.Layout();
  if (!layout.HasStrides()) {
    return "";
  }
  const auto strides = layout.Strides();
  std::vector<std::string> parts;
  parts.reserve(strides.size());
  for (uint32_t stride : strides) {
    parts.push_back(absl::StrCat(stride));
  }
  return absl::StrJoin(parts, "x");
}

absl::Status DumpTensorMap(
    absl::flat_hash_map<absl::string_view, TensorBuffer>& tensors,
    absl::string_view role, const std::filesystem::path& step_dir,
    std::ofstream& manifest) {
  for (auto& [name, tensor] : tensors) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto packed_size, tensor.PackedSize());
    const std::string filename = absl::StrCat(
        role, "__", SanitizeTensorName(name), ".csv");
    const std::filesystem::path file_path = step_dir / filename;
    LITERT_RETURN_IF_ERROR(DumpTensorToCsv(tensor, file_path.string()));
    manifest << role << '\t' << name << '\t'
             << ElementTypeName(tensor_type.ElementType()) << '\t'
             << ShapeString(tensor_type) << '\t' << BufferTypeName(tensor)
             << '\t' << packed_size << '\t' << StridesString(tensor_type)
             << '\t' << filename << '\n';
  }
  return absl::OkStatus();
}

absl::Status DumpTensorSubset(
    absl::flat_hash_map<absl::string_view, TensorBuffer>& tensors,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& tensor_names,
    absl::string_view role, const std::filesystem::path& step_dir,
    std::ofstream& manifest) {
  for (const auto& entry : tensor_names) {
    const auto name = entry.first;
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      return absl::NotFoundError(
          absl::StrCat("Missing tensor in merged decode map: ", name));
    }
    auto& tensor = it->second;
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto packed_size, tensor.PackedSize());
    const std::string filename = absl::StrCat(
        role, "__", SanitizeTensorName(name), ".csv");
    const std::filesystem::path file_path = step_dir / filename;
    LITERT_RETURN_IF_ERROR(DumpTensorToCsv(tensor, file_path.string()));
    manifest << role << '\t' << name << '\t'
             << ElementTypeName(tensor_type.ElementType()) << '\t'
             << ShapeString(tensor_type) << '\t' << BufferTypeName(tensor)
             << '\t' << packed_size << '\t' << StridesString(tensor_type)
             << '\t' << filename << '\n';
  }
  return absl::OkStatus();
}

absl::Status DumpProfilerArtifacts(
    const std::filesystem::path& summary_path, absl::string_view summary,
    absl::Span<const ProfiledEventData> events) {
  std::filesystem::create_directories(summary_path.parent_path());
  {
    std::ofstream summary_file(summary_path.string());
    if (!summary_file.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open decode profile summary file: ",
                       summary_path.string()));
    }
    summary_file << summary;
  }

  const std::filesystem::path events_path =
      std::filesystem::path(absl::StrCat(summary_path.string(), ".events.tsv"));
  std::ofstream events_file(events_path.string());
  if (!events_file.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open decode profile events file: ",
                     events_path.string()));
  }
  events_file
      << "index\ttag\tevent_type\tstart_timestamp_us\telapsed_time_us\t"
         "event_metadata1\tevent_metadata2\tevent_source\t"
         "begin_total_allocated_bytes\tend_total_allocated_bytes\n";
  for (size_t i = 0; i < events.size(); ++i) {
    const auto& event = events[i];
    events_file << i << '\t' << (event.tag ? event.tag : "") << '\t'
                << static_cast<int>(event.event_type) << '\t'
                << event.start_timestamp_us << '\t' << event.elapsed_time_us
                << '\t' << event.event_metadata1 << '\t'
                << event.event_metadata2 << '\t'
                << static_cast<int>(event.event_source) << '\t'
                << event.begin_mem_usage.total_allocated_bytes << '\t'
                << event.end_mem_usage.total_allocated_bytes << '\n';
  }
  return absl::OkStatus();
}

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr int kDynamicDimValue = -1;

// Default number of threads for WebGPU weight upload and kernel compilation.
constexpr int kDefaultNumThreadsToUpload = 2;
constexpr int kDefaultNumThreadsToCompile = 1;

int GetOutputHeads(const RuntimeConfig& runtime_config) {
  return runtime_config.output_heads.value_or(1);
}
absl::Status InitializeEmbeddingLookups(
    litert::Environment& env, ModelResources& resources,
    std::unique_ptr<EmbeddingLookupManager>& embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager>& per_layer_embedding_lookup) {
  absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
  {
    auto end_of_audio_model =
        resources.GetTFLiteModel(ModelType::kTfLiteEndOfAudio);
    if (end_of_audio_model.ok()) {
      end_of_multi_modal_embedding_models.insert(
          {ExecutorAudioData::kEndToken, end_of_audio_model.value()});
    }
  }
  {
    auto end_of_vision_model =
        resources.GetTFLiteModel(ModelType::kTfLiteEndOfVision);
    if (end_of_vision_model.ok()) {
      end_of_multi_modal_embedding_models.insert(
          {ExecutorVisionData::kEndToken, end_of_vision_model.value()});
    }
  }

  auto text_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEmbedder);
  if (text_embedder_model.ok()) {
    ASSIGN_OR_RETURN(
        embedding_lookup,
        EmbeddingLookupManager::Create(env, *text_embedder_model,
                                       end_of_multi_modal_embedding_models));
  }

  // Create per layer embedding lookups from the resources.
  auto per_layer_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder);
  if (per_layer_embedder_model.ok()) {
    ASSIGN_OR_RETURN(
        per_layer_embedding_lookup,
        EmbeddingLookupManager::Create(env, *per_layer_embedder_model,
                                       /*fully_supports_multi_modal=*/false));
  }
  return absl::OkStatus();
}

absl::Status CopyKvCacheBuffers(
    size_t decode_batch_size, int src_index_to_copy_on_prefill,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
        src_kv_cache_buffers,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
        dst_kv_cache_buffers) {
  const bool profile_kv_copy_locks = ShouldProfileKvCopyLocks();
  const absl::Time total_start_time =
      profile_kv_copy_locks ? absl::Now() : absl::Time();
  absl::Duration total_src_lock_time = absl::ZeroDuration();
  absl::Duration total_dst_lock_time = absl::ZeroDuration();
  absl::Duration total_memcpy_time = absl::ZeroDuration();
  absl::Duration total_unlock_time = absl::ZeroDuration();
  size_t total_bytes_copied = 0;
  size_t total_tensors_copied = 0;
  absl::Duration slowest_tensor_total_time = absl::ZeroDuration();
  std::string slowest_tensor_name;
  for (const auto& [name, src_buffer] : src_kv_cache_buffers) {
    if (!dst_kv_cache_buffers.contains(name)) {
      return absl::FailedPreconditionError(
          absl::StrCat("KV cache buffer ", name, " not found."));
    }
    const auto& dst_buffer = dst_kv_cache_buffers.at(name);
    const absl::Time tensor_start_time =
        profile_kv_copy_locks ? absl::Now() : absl::Time();
    absl::Time unlock_start_time;
    size_t copied_bytes_for_tensor = 0;
    {
      const absl::Time src_lock_start_time =
          profile_kv_copy_locks ? absl::Now() : absl::Time();
      LITERT_ASSIGN_OR_RETURN(auto src_buffer_lock_and_addr,
                              TensorBufferScopedLock::Create(
                                  src_buffer, TensorBuffer::LockMode::kRead));
      if (profile_kv_copy_locks) {
        total_src_lock_time += absl::Now() - src_lock_start_time;
      }
      LITERT_ASSIGN_OR_RETURN(size_t src_buffer_size, src_buffer.PackedSize());
      const char* src_buffer_ptr =
          static_cast<const char*>(src_buffer_lock_and_addr.second);

      const absl::Time dst_lock_start_time =
          profile_kv_copy_locks ? absl::Now() : absl::Time();
      LITERT_ASSIGN_OR_RETURN(auto dst_buffer_lock_and_addr,
                              TensorBufferScopedLock::Create(
                                  dst_buffer, TensorBuffer::LockMode::kWrite));
      if (profile_kv_copy_locks) {
        total_dst_lock_time += absl::Now() - dst_lock_start_time;
      }
      LITERT_ASSIGN_OR_RETURN(size_t dst_buffer_size, dst_buffer.PackedSize());
      char* dst_buffer_ptr = static_cast<char*>(
          const_cast<void*>(dst_buffer_lock_and_addr.second));
      // This copy is based on the assumption that the KV cache buffers are in
      // the layout of [batch * X, ...] or [1, batch * X, ...] where X could be
      // 1 or more and X doesn't make values interleaved across batches which is
      // true for the current LLM models of all backends.
      const absl::Time memcpy_start_time =
          profile_kv_copy_locks ? absl::Now() : absl::Time();
      if (src_index_to_copy_on_prefill >= 0) {
        // This is the case of the first prefill after decode. It reduces the KV
        // cache size to one by copying only the cache content of the given
        // index.
        RET_CHECK_EQ(src_buffer_size, dst_buffer_size * decode_batch_size);
        RET_CHECK_LT(src_index_to_copy_on_prefill, decode_batch_size);
        src_buffer_ptr += src_index_to_copy_on_prefill * dst_buffer_size;
        memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
        copied_bytes_for_tensor = dst_buffer_size;
      } else {
        // This is the case of the first decode after prefill. It broadcasts the
        // KV cache contents to all the batches.
        RET_CHECK_EQ(src_buffer_size * decode_batch_size, dst_buffer_size);
        for (int i = 0; i < decode_batch_size; ++i) {
          memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
          dst_buffer_ptr += src_buffer_size;
        }
        copied_bytes_for_tensor = src_buffer_size * decode_batch_size;
      }
      if (profile_kv_copy_locks) {
        total_memcpy_time += absl::Now() - memcpy_start_time;
        unlock_start_time = absl::Now();
      }
    }
    if (profile_kv_copy_locks) {
      total_unlock_time += absl::Now() - unlock_start_time;
      total_bytes_copied += copied_bytes_for_tensor;
      ++total_tensors_copied;
      const absl::Duration tensor_total_time = absl::Now() - tensor_start_time;
      if (tensor_total_time > slowest_tensor_total_time) {
        slowest_tensor_total_time = tensor_total_time;
        slowest_tensor_name = std::string(name);
      }
    }
  }
  if (profile_kv_copy_locks) {
    const absl::Duration total_elapsed = absl::Now() - total_start_time;
    const double mib_copied =
        static_cast<double>(total_bytes_copied) / (1024.0 * 1024.0);
    const double seconds = absl::ToDoubleSeconds(total_elapsed);
    const double throughput_mib_per_s =
        seconds > 0.0 ? mib_copied / seconds : 0.0;
    ABSL_LOG(INFO)
        << "CopyKvCacheBuffers profile"
        << " tensors=" << total_tensors_copied
        << " bytes=" << total_bytes_copied
        << " mib=" << mib_copied
        << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed)
        << " src_lock_ms=" << absl::ToDoubleMilliseconds(total_src_lock_time)
        << " dst_lock_ms=" << absl::ToDoubleMilliseconds(total_dst_lock_time)
        << " memcpy_ms=" << absl::ToDoubleMilliseconds(total_memcpy_time)
        << " unlock_ms=" << absl::ToDoubleMilliseconds(total_unlock_time)
        << " throughput_mib_per_s=" << throughput_mib_per_s
        << " slowest_tensor=" << slowest_tensor_name
        << " slowest_tensor_total_ms="
        << absl::ToDoubleMilliseconds(slowest_tensor_total_time);
  }
  return absl::OkStatus();
}

absl::StatusOr<int> GetDynamicDimIndex(const Model& model,
                                       absl::string_view signature,
                                       absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& sig,
                          model.FindSignature(signature));
  LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                          sig.InputTensor(tensor_name));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      return i;
    }
  }
  return absl::InvalidArgumentError("No dynamic dimension found.");
}

absl::StatusOr<bool> HasDynamicDim(const Model& model,
                                   absl::string_view signature,
                                   absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& sig,
                          model.FindSignature(signature));
  LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                          sig.InputTensor(tensor_name));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      return true;
    }
  }
  return false;
}

absl::Status ResolveDynamicShape(const Model& model,
                                 CompiledModel& compiled_model,
                                 absl::string_view signature,
                                 absl::string_view tensor_name, int new_value) {
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& sig,
                          model.FindSignature(signature));
  LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                          sig.InputTensor(tensor_name));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
  auto dimensions = ranked_tensor_type.Layout().Dimensions();

  bool has_dynamic_dim = false;
  std::vector<int> new_shape;
  new_shape.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      has_dynamic_dim = true;
      new_shape.push_back(new_value);
    } else {
      new_shape.push_back(dimensions[i]);
    }
  }

  if (has_dynamic_dim) {
    LITERT_RETURN_IF_ERROR(
        compiled_model.ResizeInputTensor(signature, tensor_name, new_shape));
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> ResizeKVCacheTensorBuffer(
    Environment& env, TensorBuffer& tensor_buffer, int dynamic_dim_index,
    int num_entries_to_insert) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType& tensor_type,
                          tensor_buffer.TensorType());
  RET_CHECK(!tensor_type.Layout().HasStrides());
  auto dimensions = tensor_type.Layout().Dimensions();
  std::vector<int> new_dimensions;
  new_dimensions.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (i == dynamic_dim_index) {
      new_dimensions.push_back(dimensions[i] + num_entries_to_insert);
    } else {
      new_dimensions.push_back(dimensions[i]);
    }
  }

  LITERT_ASSIGN_OR_RETURN(TensorBufferType buffer_type,
                          tensor_buffer.BufferType());
  Layout new_layout(Dimensions(new_dimensions.begin(), new_dimensions.end()));
  auto new_out_type =
      RankedTensorType(tensor_type.ElementType(), std::move(new_layout));
  LITERT_ASSIGN_OR_RETURN(size_t new_size, new_out_type.Bytes());

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer new_tensor_buffer,
      TensorBuffer::CreateManaged(env, buffer_type, new_out_type, new_size));

  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  auto* tensor_buffer_ptr =
      static_cast<uint8_t*>(tensor_buffer_lock_and_addr.second);
  LITERT_ASSIGN_OR_RETURN(
      auto new_tensor_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(new_tensor_buffer,
                                     TensorBuffer::LockMode::kWrite));
  auto* new_tensor_buffer_ptr =
      static_cast<uint8_t*>(new_tensor_buffer_lock_and_addr.second);
  std::optional<size_t> element_size = GetByteWidth(tensor_type.ElementType());
  RET_CHECK(element_size.has_value());

  RETURN_IF_ERROR(ExpandBuffer(tensor_buffer_ptr, dimensions,
                               new_tensor_buffer_ptr, new_dimensions,
                               element_size.value()));

  return new_tensor_buffer;
}

// Builds the output tensor type for the embedding lookup. The output tensor
// type is the same as the input tensor type, except the first dimension is the
// number of tokens.
absl::StatusOr<RankedTensorType> GetEmbeddingLookupOutputTensorType(
    int num_tokens, const RankedTensorType& output_element_type) {
  if (num_tokens == 1) {
    return output_element_type;
  } else if (num_tokens == 0) {
    return absl::InvalidArgumentError(
        "Number of tokens must be greater than 0.");
  }

  const auto& dims = output_element_type.Layout().Dimensions();
  if (dims.size() < 3) {
    return absl::InvalidArgumentError("Tensor type must have rank 3 or more.");
  }
  if (dims[0] != 1 || dims[1] != 1) {
    return absl::InvalidArgumentError(
        "Element type must have first two dimensions as 1.");
  }
  Dimensions embedding_dims(dims.begin(), dims.end());
  embedding_dims[1] = num_tokens;
  return RankedTensorType(output_element_type.ElementType(),
                          Layout(std::move(embedding_dims)));
}

struct MaybeWrappedTensorBuffer {
  TensorBuffer buffer;
  bool wrapped;
};

template <typename T>
absl::StatusOr<MaybeWrappedTensorBuffer> WrapOrCreateTensorBufferFromHostMemory(
    RankedTensorType tensor_type, absl::Span<T> data) {
  size_t size = data.size() * sizeof(T);
  // First try to wrap the memory with a TensorBuffer.
  auto wrapped_buffer =
      TensorBuffer::CreateFromHostMemory(tensor_type, data.data(), size);
  if (wrapped_buffer.HasValue()) {
    return MaybeWrappedTensorBuffer{.buffer = std::move(*wrapped_buffer),
                                    .wrapped = true};
  }

  LITERT_ASSIGN_OR_RETURN(
      auto new_buffer,
      TensorBuffer::CreateManagedHostMemory(tensor_type, size));
  return MaybeWrappedTensorBuffer{.buffer = std::move(new_buffer),
                                  .wrapped = false};
}

// Returns a subspan of the given span for a chunk at the given index.
template <typename T>
absl::Span<const T> GetSpanForChunk(absl::Span<T> span, int num_chunks,
                                    int chunk_index) {
  size_t total_size = span.size();
  size_t chunk_size = total_size / num_chunks;
  return span.subspan(chunk_size * chunk_index, chunk_size);
}

absl::StatusOr<TensorBuffer> CreateFP16OutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    absl::string_view output_name, size_t output_index) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  // Use runtime layout.
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(
      auto requirements,
      compiled_model.GetOutputBufferRequirements(signature_index, output_name));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType new_tensor_type(litert::ElementType::Float16,
                                   std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer_types, requirements.SupportedTypes());
  if (buffer_types.empty()) {
    return absl::InternalError("No supported buffer types found.");
  }
  auto buffer_type = buffer_types[0];
  LITERT_ASSIGN_OR_RETURN(
      auto buffer, TensorBuffer::CreateManaged(
                       env, buffer_type, std::move(new_tensor_type), size));
  return buffer;
}

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutorBase::CreatePrefillInputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    int context_length,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_input_buffers) {
  BenchmarkMark(benchmark_info_, "llm_prefill_create_buffers");
  auto dyn_shape_resolver = [&](absl::string_view tensor_name) -> absl::Status {
    return ResolveDynamicShape(model_, compiled_model_, prefill_signature,
                               tensor_name, sequence_length);
  };
  // Create input_token, positions and attn_mask buffers after determining
  // the prefill length.
  if (!signatures_.input_tokens.empty()) {
    RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_tokens));
    auto tokens_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_tokens);
    prefill_input_buffers[signatures_.input_tokens] = std::move(*tokens_buffer);
  } else {
    // If input_tokens is empty, we must have input_embeddings.
    if (!signatures_.input_embeddings.has_value()) {
      return absl::FailedPreconditionError(
          "Input tokens or embeddings must be provided.");
    }
    if (embedding_lookup_ == nullptr) {
      return absl::FailedPreconditionError(
          "Input embeddings required by signature but embedding lookup "
          "model is not initialized.");
    }
    RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_embeddings.value()));
    auto embeddings_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_embeddings.value());
    prefill_input_buffers[signatures_.input_embeddings.value()] =
        std::move(*embeddings_buffer);

    // We may have per layer embedding as well.
    if (signatures_.input_per_layer_embeddings.has_value()) {
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Input per layer embeddings required by signature but "
            "embedding lookup model is not initialized.");
      }
      RETURN_IF_ERROR(
          dyn_shape_resolver(signatures_.input_per_layer_embeddings.value()));
      auto per_layer_embeddings_buffer = compiled_model_.CreateInputBuffer(
          prefill_signature, signatures_.input_per_layer_embeddings.value());
      prefill_input_buffers[signatures_.input_per_layer_embeddings.value()] =
          std::move(*per_layer_embeddings_buffer);
    }
  }
  RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_positions));
  auto positions_buffer = compiled_model_.CreateInputBuffer(
      prefill_signature, signatures_.input_positions);
  if (ShouldDebugInputPosLock() && positions_buffer.HasValue()) {
    ABSL_LOG(INFO) << "Prefill input_pos buffer created signature="
                   << prefill_signature
                   << " name=" << signatures_.input_positions << " "
                   << TensorBufferSummary(*positions_buffer);
  }
  prefill_input_buffers[signatures_.input_positions] =
      std::move(*positions_buffer);

  if (signatures_.input_attn_mask.has_value()) {
    ASSIGN_OR_RETURN(bool is_attn_dyn,
                     HasDynamicDim(model_, prefill_signature,
                                   signatures_.input_attn_mask.value()));
    if (is_attn_dyn) {
      std::vector<int> new_shape = {1, 1, sequence_length, context_length};
      LITERT_RETURN_IF_ERROR(compiled_model_.ResizeInputTensor(
          prefill_signature, signatures_.input_attn_mask.value(), new_shape));
    }

    auto attn_mask_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_attn_mask.value());
    prefill_input_buffers[signatures_.input_attn_mask.value()] =
        std::move(*attn_mask_buffer);
  }
  if (signatures_.input_int32_param.has_value()) {
    gpu_optimized_single_buffer_cache_ = true;
    auto param_tensor_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_int32_param.value());
    prefill_input_buffers[signatures_.input_int32_param.value()] =
        std::move(*param_tensor_buffer);
  }
  BenchmarkMark(benchmark_info_, "llm_prefill_create_buffers");
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::FillInputBufferWithToken(
    const std::vector<std::shared_ptr<TokenData>>& unprocessed_token,
    TensorBuffer& input_buffer, bool is_per_layer_embedding) {
  if (unprocessed_token.empty()) {
    return absl::InvalidArgumentError("Unprocessed token is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto input_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              input_buffer, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t packed_size, input_buffer.PackedSize());
  size_t stride = packed_size / unprocessed_token.size();
  char* input_buffer_ptr =
      static_cast<char*>(input_buffer_lock_and_addr.second);
  for (const auto& token : unprocessed_token) {
    size_t size_to_fill = 0;
    if (token->embedding().empty()) {
      size_to_fill = sizeof(int32_t);
      RET_CHECK_GE(stride, size_to_fill);
      // If the token has no embedding, the input_buffer should takes token id.
      *reinterpret_cast<int32_t*>(input_buffer_ptr) = token->id();
    } else if (is_per_layer_embedding) {
      size_to_fill = token->per_layer_embedding().size() * sizeof(float);
      RET_CHECK_GE(stride, size_to_fill);
      memcpy(input_buffer_ptr, token->per_layer_embedding().data(),
             size_to_fill);
    } else {
      size_to_fill = token->embedding().size() * sizeof(float);
      RET_CHECK_GE(stride, size_to_fill);
      memcpy(input_buffer_ptr, token->embedding().data(), size_to_fill);
    }

    if (stride > size_to_fill) {
      memset(input_buffer_ptr + size_to_fill, 0, stride - size_to_fill);
    }
    input_buffer_ptr += stride;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RollBackProcessedTokens() {
  int current_step = llm_context_->runtime_state().current_step;
  ProcessedTokens& processed_tokens =
      llm_context_->processed_context().processed_tokens();
  if (current_step == processed_tokens.TokenCount()) {
    return absl::OkStatus();
  }
  if (current_step == 0) {
    RETURN_IF_ERROR(processed_tokens.RollBackToStep(0));
  } else {
    auto token_at_step = processed_tokens.GetTokenAtStep(current_step - 1);
    RETURN_IF_ERROR(processed_tokens.RollBackToStep(current_step - 1));
    if (!token_at_step.empty()) {
      RET_CHECK_EQ(token_at_step.size(), 1);
      // Multimodal input cannot become a pending input token.
      if (token_at_step.at(0) > 0) {
        RETURN_IF_ERROR(processed_tokens.AddPendingInputToken(
            {std::make_shared<TokenData>(token_at_step.at(0))}));
      } else {
        processed_tokens.AddProcessedTokens({token_at_step.at(0)});
      }
    }
  }

  // Reset sampler input handling as the step is rolled back.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrepareFirstPrefillAfterDecode(
    int token_index_to_reduce) {
  if (!llm_context_->runtime_state().ran_decode && !force_prepare_needed_) {
    return absl::OkStatus();
  }

  force_prepare_needed_ = false;
  llm_context_->runtime_state().ran_decode = false;

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  if (output_heads > 1) {
    LITERT_RETURN_IF_ERROR(llm_context_->processed_context()
                               .processed_tokens()
                               .ReduceTokenCandidates(token_index_to_reduce));
    LITERT_RETURN_IF_ERROR(
        CopyKvCacheBuffers(output_heads, token_index_to_reduce,
                           *input_kv_cache_buffers_, kv_cache_buffers_1_));
    input_kv_cache_buffers_ = &kv_cache_buffers_1_;
    output_kv_cache_buffers_ = &kv_cache_buffers_2_;
  }

  // Reset sampler input handling if it handles input for next decode.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    Span<const int> ids, bool async) {
  const int trace_step = llm_context_->runtime_state().current_step;
  TracePhaseBegin(trace_step, kTracePrefillPrepareInputs);
  BenchmarkMark(benchmark_info_, "llm_prefill_prepare_inputs");
  RETURN_IF_ERROR(RollBackProcessedTokens());

  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_pos =
        prefill_input_buffers[signatures_.input_positions];
    if (ShouldDebugInputPosLock()) {
      ABSL_LOG(INFO) << "Prefill input_pos before lock signature="
                     << prefill_signature
                     << " name=" << signatures_.input_positions << " "
                     << TensorBufferSummary(prefill_input_pos);
    }
    LITERT_ASSIGN_OR_RETURN(auto prefill_input_pos_size,
                            prefill_input_pos.PackedSize());
    auto prefill_input_pos_lock_and_addr = TensorBufferScopedLock::Create(
        prefill_input_pos, TensorBuffer::LockMode::kWrite);
    if (!prefill_input_pos_lock_and_addr.HasValue()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to lock prefill input_pos for write. "
                       "signature=",
                       prefill_signature, " name=",
                       signatures_.input_positions, " summary=",
                       TensorBufferSummary(prefill_input_pos), " lock_error=",
                       prefill_input_pos_lock_and_addr.Error().Message()));
    }
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr->second);

    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    if (signatures_.input_attn_mask.has_value()) {
      RETURN_IF_ERROR(InitializeAttentionMask(
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          use_fp16_precision_));
    }
    // TODO(b/425396146): Add the unit tests for checking the prefill length.
    // We always hold one pending token in the input ids for the next
    // prefill or decode step.
    int prefill_length = ids.size() - 1;

    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        llm_context_->processed_context()
            .processed_tokens()
            .GetNextUnprocessedToken();
    RET_CHECK_LE(pending_input_token.size(), 1);
    const int start_step = internal_start_step;
    const bool has_pending_input_token = !pending_input_token.empty();
    const bool use_token_as_lookup = !signatures_.input_tokens.empty();
    const bool use_per_layer_embedding =
        signatures_.input_per_layer_embeddings.has_value();
    // If there is no pending input token and no input token to prefill, we can
    // skip the prefill by storing the token as a pending input token.
    bool skip_prefill = !has_pending_input_token && prefill_length == 0;
    if (!skip_prefill) {
      int input_idx = 0;
      if (has_pending_input_token) {
        if (use_token_as_lookup) {
          RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_tokens]));
        } else {
          RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_embeddings.value()]));
          if (use_per_layer_embedding) {
            RETURN_IF_ERROR(FillInputBufferWithToken(
                pending_input_token,
                prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()],
                /*is_per_layer_embedding=*/true));
          }
        }
        prefill_input_pos_ptr[input_idx] = internal_start_step;
        RETURN_IF_ERROR(llm_context_->processed_context()
                            .processed_tokens()
                            .MarkPendingInputTokenAsProcessed());
        ++prefill_input_pos_ptr;
        ++input_idx;
      }
      std::transform(prefill_input_pos_ptr,
                     prefill_input_pos_ptr + prefill_length,
                     prefill_input_pos_ptr, [&](int token) mutable {
                       return llm_context_->runtime_state().current_step++;
                     });
      std::vector<int> processed_input_tokens(ids.begin(),
                                              ids.begin() + prefill_length);
      llm_context_->processed_context().processed_tokens().AddProcessedTokens(
          processed_input_tokens);

      if (use_token_as_lookup) {
        auto& prefill_input_buffer =
            prefill_input_buffers[signatures_.input_tokens];
        LITERT_ASSIGN_OR_RETURN(
            auto prefill_input_lock_and_addr,
            TensorBufferScopedLock::Create(prefill_input_buffer,
                                           TensorBuffer::LockMode::kWrite));
        int32_t* prefill_input_ptr =
            static_cast<int32_t*>(prefill_input_lock_and_addr.second);
        if (!has_pending_input_token) {
          LITERT_ASSIGN_OR_RETURN(auto prefill_input_size,
                                  prefill_input_buffer.PackedSize());
          // If there is a pending input token, the zeros and the pending input
          // token id are already filled in the above
          // FillInputBufferWithToken() function, so we cannot zero out the
          // whole prefill input buffer here.
          //
          // If there is no pending input token, we need to zero out the whole
          // prefill input buffer.
          memset(prefill_input_ptr, 0, prefill_input_size);
        }
        memcpy(prefill_input_ptr + input_idx, processed_input_tokens.data(),
               processed_input_tokens.size() * sizeof(int32_t));
      } else {
        // If not using token as lookup, we must have input_embeddings. There is
        // no need to create input_embeddings_ptr because TensorBuffer locking
        // and filling is handled by the embedding lookup.
        TensorBuffer* prefill_input_embeddings_buffer =
            &(prefill_input_buffers[signatures_.input_embeddings.value()]);
        TracePhaseBegin(trace_step, kTracePrefillEmbeddingLookup);
        BenchmarkMark(benchmark_info_, "llm_prefill_embedding_lookup");
        RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
            processed_input_tokens, prefill_input_embeddings_buffer,
            /*offset=*/input_idx));

        // We may have per layer embedding as well.
        if (signatures_.input_per_layer_embeddings) {
          TensorBuffer* prefill_input_per_layer_embeddings_buffer =
              &(prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()]);
          RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
              processed_input_tokens, prefill_input_per_layer_embeddings_buffer,
              /*offset=*/input_idx));
        }
        BenchmarkMark(benchmark_info_, "llm_prefill_embedding_lookup");
        TracePhaseEnd(trace_step, kTracePrefillEmbeddingLookup);
      }
      if (signatures_.input_attn_mask.has_value()) {
        RETURN_IF_ERROR(FillAttentionMask(
            prefill_input_buffers[signatures_.input_attn_mask.value()],
            start_step,
            /*steps=*/prefill_length + input_idx));
      }
      if (gpu_optimized_single_buffer_cache_) {
        LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
        RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
            prefill_input_buffers[signatures_.input_int32_param.value()],
            start_step, ids.size()));
      }
    }

    // Add the last token of the current input as a pending input token, to be
    // used in the next prefill or decode.
    auto last_input_token = std::make_shared<TokenData>(ids.back());
    if (!use_token_as_lookup) {
      // Look up the embeddings for the last token so they can be used in the
      // next prefill or decode. This has to be done now in the case of
      // multi-modal prefill so the embeddings are used in the correct order.
      BenchmarkMark(benchmark_info_, "llm_prefill_embedding_lookup");
      TracePhaseBegin(trace_step, kTracePrefillEmbeddingLookup);
      RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          last_input_token->id(), last_input_token->mutable_embedding()));
      if (use_per_layer_embedding) {
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            last_input_token->id(),
            last_input_token->mutable_per_layer_embedding()));
      }
      BenchmarkMark(benchmark_info_, "llm_prefill_embedding_lookup");
      TracePhaseEnd(trace_step, kTracePrefillEmbeddingLookup);
    }
    // Add the last input token to the pending input token list.
    RETURN_IF_ERROR(llm_context_->processed_context()
                        .processed_tokens()
                        .AddPendingInputToken({std::move(last_input_token)}));
    ++llm_context_->runtime_state().current_step;
    if (skip_prefill) {
      BenchmarkMark(benchmark_info_, "llm_prefill_prepare_inputs");
      TracePhaseEnd(trace_step, kTracePrefillPrepareInputs);
      return absl::OkStatus();
    }
  }
  BenchmarkMark(benchmark_info_, "llm_prefill_prepare_inputs");
  TracePhaseEnd(trace_step, kTracePrefillPrepareInputs);
  return BindTensorsAndRunPrefill(prefill_signature, prefill_input_buffers,
                                  async);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunPrefill(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    bool async) {
  const int trace_step = llm_context_->runtime_state().current_step - 1;
  TracePhaseBegin(trace_step, kTracePrefillBindBuffers);
  BenchmarkMark(benchmark_info_, "llm_prefill_bind_buffers");
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
  for (const auto& [input_name, input_buffer] : prefill_input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    input_buffers[input_name] = std::move(input_buffer_dup);
  }
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    output_buffers[output_name] = std::move(output_buffer_dup);
  }
  LITERT_ASSIGN_OR_RETURN(auto expected_input_names,
                          compiled_model_.GetSignatureInputNames(
                              prefill_signature));
  for (absl::string_view input_name : expected_input_names) {
    if (!input_buffers.contains(input_name)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing prefill input buffer for signature '", prefill_signature,
          "': ", input_name));
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto expected_output_names,
                          compiled_model_.GetSignatureOutputNames(
                              prefill_signature));
  for (absl::string_view output_name : expected_output_names) {
    if (!output_buffers.contains(output_name)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing prefill output buffer for signature '", prefill_signature,
          "': ", output_name));
    }
  }
  BenchmarkMark(benchmark_info_, "llm_prefill_bind_buffers");
  TracePhaseEnd(trace_step, kTracePrefillBindBuffers);

  TracePhaseBegin(trace_step, kTracePrefillModelRun);
  BenchmarkMark(benchmark_info_, "llm_prefill_model_run");
  if (async) {
    LITERT_RETURN_IF_ERROR(compiled_model_.RunAsync(
        prefill_signature, input_buffers, output_buffers, async));
  } else {
    LITERT_RETURN_IF_ERROR(
        compiled_model_.Run(prefill_signature, input_buffers, output_buffers));
  }
  BenchmarkMark(benchmark_info_, "llm_prefill_model_run");
  TracePhaseEnd(trace_step, kTracePrefillModelRun);

  if (!gpu_optimized_single_buffer_cache_) {
    std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  }
  return absl::OkStatus();
}

absl::StatusOr<ProcessedTokens::StepAndToken>
LlmLiteRtCompiledModelExecutorBase::GetTokenToDecode(
    const ExecutorInputs& inputs) {
  const int step = llm_context_->runtime_state().current_step - 1;
  TracePhaseBegin(step, kTraceDecodePrepareInputs);
  BenchmarkMark(benchmark_info_, "llm_decode_prepare_inputs");
  RETURN_IF_ERROR(RollBackProcessedTokens());

  if (inputs.GetTextDataPtr().ok()) {
    LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
    auto input_tensor_size = token_ids_buffer->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      // Input token ids provided, so use it regardless of whether next input
      // token id is set.
      RET_CHECK_EQ(*input_tensor_size, output_heads * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN(
          auto ids, ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));
      if (ids[0] >= 0) {
        // If the input token id is >= 0, it means the input token is provided
        // by the user. In this case, we should invalidate the pending input
        // token and add the input token as a pending input token.
        llm_context_->processed_context()
            .processed_tokens()
            .InvalidatePendingInputToken();
        std::vector<std::shared_ptr<TokenData>> token;
        token.reserve(output_heads);
        for (int i = 0; i < output_heads; ++i) {
          token.push_back(std::make_shared<TokenData>(ids[i]));
        }
        RETURN_IF_ERROR(llm_context_->processed_context()
                            .processed_tokens()
                            .AddPendingInputToken(token));
      }
    }
  }

  // Here we must have a pending input token to decode that's either coming from
  // the previous prefill or decode, or we just added one from the inputs.
  for (const auto& token : llm_context_->processed_context()
                               .processed_tokens()
                               .GetNextUnprocessedToken()
                               .token) {
    // If the token has no embedding, we will look up the embedding for the
    // token here. This reduces the complexity for internal or external
    // sampling.
    if (signatures_.input_embeddings.has_value() &&
        token->mutable_embedding().empty()) {
      TracePhaseBegin(step, kTraceDecodeEmbeddingLookup);
      BenchmarkMark(benchmark_info_, "llm_decode_embedding_lookup");
      RETURN_IF_ERROR(embedding_lookup_->LookupDecode(
          token->id(), token->mutable_embedding()));
      if (signatures_.input_per_layer_embeddings.has_value()) {
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
            token->id(), token->mutable_per_layer_embedding()));
      }
      BenchmarkMark(benchmark_info_, "llm_decode_embedding_lookup");
      TracePhaseEnd(step, kTraceDecodeEmbeddingLookup);
    }
  }
  BenchmarkMark(benchmark_info_, "llm_decode_prepare_inputs");
  TracePhaseEnd(step, kTraceDecodePrepareInputs);
  return llm_context_->processed_context()
      .processed_tokens()
      .GetNextUnprocessedToken();
}

absl::Status
LlmLiteRtCompiledModelExecutorBase::ConsumePendingOrAddProcessedToken(
    const std::vector<std::shared_ptr<TokenData>>& token) {
  auto status = llm_context_->processed_context()
                    .processed_tokens()
                    .MarkPendingInputTokenAsProcessed();
  if (status.ok() || status.code() != absl::StatusCode::kNotFound) {
    return status;
  }

  // If the pending input token was not used, we should add the token to the
  // processed tokens.
  std::vector<int> processed_tokens;
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  processed_tokens.reserve(output_heads);
  for (const auto& t : token) {
    processed_tokens.push_back(t->id());
  }
  llm_context_->processed_context().processed_tokens().AddProcessedTokens(
      processed_tokens);
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  int step = llm_context_->runtime_state().current_step - 1;
  if (sampler_ && sampler_->HandlesInput()) {
    // The sampler has already been running decode for this step. Check if
    // output_logits is the one used last time, i.e. by
    // BindTensorsAndRunDecodeStatic().
    LITERT_RETURN_IF_ERROR(
        output_logits.Get() ==
        decode_output_buffers_[signatures_.output_logits].Get());
    return absl::OkStatus();
  }

  const bool use_token_as_lookup = !signatures_.input_tokens.empty();
  const bool use_per_layer_embedding =
      signatures_.input_per_layer_embeddings.has_value();

  // Fill the input buffers with scoped locks.
  BenchmarkMark(benchmark_info_, "llm_decode_prepare_inputs");
  if (use_token_as_lookup) {
    RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_tokens]));
  } else {
    if (!signatures_.input_embeddings.has_value()) {
      return absl::InvalidArgumentError(
          "Input tokens or embeddings must be provided.");
    }
    RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_embeddings.value()]));
    if (use_per_layer_embedding) {
      RETURN_IF_ERROR(FillInputBufferWithToken(
          token,
          decode_input_buffers_[signatures_.input_per_layer_embeddings.value()],
          /*is_per_layer_embedding=*/true));
    }
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_type,
        decode_input_buffers_[signatures_.input_positions].TensorType());
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(
            decode_input_buffers_[signatures_.input_positions],
            TensorBuffer::LockMode::kWrite));
    auto* input_pos_ptr = static_cast<int32_t*>(input_pos_lock_and_addr.second);
    if (input_pos_type.Layout().Dimensions()[0] == 1) {
      *input_pos_ptr = step;
    } else {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      RET_CHECK_EQ(input_pos_type.Layout().Dimensions()[0], output_heads);
      LITERT_ASSIGN_OR_RETURN(
          auto input_pos_size,
          decode_input_buffers_[signatures_.input_positions].PackedSize());
      size_t offset = input_pos_size / output_heads / sizeof(int32_t);
      for (int i = 0; i < output_heads; ++i) {
        input_pos_ptr[i * offset] = step;
      }
    }
  }

  if (signatures_.input_attn_mask.has_value()) {
    RETURN_IF_ERROR(InitializeAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        use_fp16_precision_));
    RETURN_IF_ERROR(FillAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()], step,
        /*steps=*/1));
  }
  if (gpu_optimized_single_buffer_cache_) {
    LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
    RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        decode_input_buffers_[signatures_.input_int32_param.value()], step, 1));
  }
  BenchmarkMark(benchmark_info_, "llm_decode_prepare_inputs");

  return BindTensorsAndRunDecode(&output_logits);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecode(
    TensorBuffer* output_logits) {
  const int step = llm_context_->runtime_state().current_step - 1;
  TracePhaseBegin(step, kTraceDecodeBindBuffers);
  BenchmarkMark(benchmark_info_, "llm_decode_bind_buffers");
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    // LITERT_ASSIGN_OR_RETURN() causes a compilation error on windows.
    auto output_buffer_dup =
        output_logits && output_name == signatures_.output_logits
            ? output_logits->Duplicate()
            : output_buffer.Duplicate();
    RET_CHECK(output_buffer_dup) << "Failed to duplicate output buffer.";
    output_buffer_dup->ClearEvent();
    decode_output_buffers[output_name] = std::move(*output_buffer_dup);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    decode_output_buffers[output_name] = std::move(output_buffer_dup);
  }
  BenchmarkMark(benchmark_info_, "llm_decode_bind_buffers");
  TracePhaseEnd(step, kTraceDecodeBindBuffers);

  std::optional<std::filesystem::path> dump_step_dir;
  const auto& advanced_settings = executor_settings_.GetAdvancedSettings();
  const std::string dump_first_decode_tensors_dir =
      advanced_settings.has_value() &&
              !advanced_settings->dump_first_decode_tensors_dir.empty()
          ? advanced_settings->dump_first_decode_tensors_dir
          : GetFirstDecodeDumpDirFromEnv();
  if (!dumped_first_decode_tensors_ &&
      !dump_first_decode_tensors_dir.empty()) {
    const int step = llm_context_->runtime_state().current_step - 1;
    dump_step_dir =
        std::filesystem::path(dump_first_decode_tensors_dir) /
        absl::StrCat("step_", step);
    std::filesystem::create_directories(*dump_step_dir);
    std::ofstream manifest((*dump_step_dir / "manifest.tsv").string());
    if (!manifest.is_open()) {
      return absl::InternalError(absl::StrCat(
          "Failed to open first-decode manifest file under ",
          dump_step_dir->string()));
    }
    manifest << "role\tname\tdtype\tshape\tbuffer_type\tpacked_size\tstrides\tcsv_path\n";
    LITERT_RETURN_IF_ERROR(DumpTensorMap(
        decode_input_buffers_, "decode_input", *dump_step_dir, manifest));
    LITERT_RETURN_IF_ERROR(DumpTensorMap(
        *input_kv_cache_buffers_, "kv_input", *dump_step_dir, manifest));
  }

  std::optional<litert::Profiler> decode_profiler;
  const std::string dump_first_decode_profile_path =
      advanced_settings.has_value() &&
              !advanced_settings->dump_first_decode_profile_path.empty()
          ? advanced_settings->dump_first_decode_profile_path
          : GetFirstDecodeProfilePathFromEnv();
  if (!dumped_first_decode_profile_ &&
      !dump_first_decode_profile_path.empty()) {
    LiteRtProfiler profiler_handle = nullptr;
    const auto profiler_status = compiled_model_.env_.runtime
                                     ->CompiledModelGetProfiler(
                                         compiled_model_.Get(),
                                         &profiler_handle);
    if (profiler_status != kLiteRtStatusOk) {
      ABSL_LOG(WARNING) << "Failed to get decode profiler: "
                        << static_cast<int>(profiler_status);
    } else {
      decode_profiler.emplace(profiler_handle, OwnHandle::kNo);
      if (auto status = decode_profiler->Reset(); !status.HasValue()) {
        ABSL_LOG(WARNING) << "Failed to reset decode profiler: "
                          << status.Error().Message();
        decode_profiler.reset();
      } else if (auto status = decode_profiler->StartProfiling();
                 !status.HasValue()) {
        ABSL_LOG(WARNING) << "Failed to start decode profiler: "
                          << status.Error().Message();
        decode_profiler.reset();
      }
    }
  }

  bool async = true;
  TracePhaseBegin(step, kTraceDecodeModelRun);
  BenchmarkMark(benchmark_info_, "llm_decode_model_run");
  LITERT_RETURN_IF_ERROR(
      compiled_model_.RunAsync(kDecodeSignatureRunner, decode_input_buffers,
                               decode_output_buffers, async));
  BenchmarkMark(benchmark_info_, "llm_decode_model_run");
  TracePhaseEnd(step, kTraceDecodeModelRun);

  if (decode_profiler.has_value()) {
    if (auto status = decode_profiler->StopProfiling(); !status.HasValue()) {
      ABSL_LOG(WARNING) << "Failed to stop decode profiler: "
                        << status.Error().Message();
    } else {
      auto summary =
          decode_profiler->GetProfileSummary(compiled_model_.Get());
      auto events = decode_profiler->GetEvents();
      if (!summary.HasValue()) {
        ABSL_LOG(WARNING) << "Failed to get decode profile summary: "
                          << summary.Error().Message();
      } else if (!events.HasValue()) {
        ABSL_LOG(WARNING) << "Failed to get decode profile events: "
                          << events.Error().Message();
      } else {
        const std::filesystem::path summary_path(
            dump_first_decode_profile_path);
        if (auto dump_status =
                DumpProfilerArtifacts(summary_path, *summary, *events);
            !dump_status.ok()) {
          ABSL_LOG(WARNING) << "Failed to dump decode profile artifacts: "
                            << dump_status;
        } else {
          dumped_first_decode_profile_ = true;
        }
      }
    }
  }

  if (dump_step_dir.has_value()) {
    std::ofstream manifest(
        ((*dump_step_dir) / "manifest.tsv").string(),
        std::ios::app);
    if (!manifest.is_open()) {
      return absl::InternalError(absl::StrCat(
          "Failed to reopen first-decode manifest file under ",
          dump_step_dir->string()));
    }
    LITERT_RETURN_IF_ERROR(DumpTensorSubset(
        decode_output_buffers, decode_output_buffers_, "decode_output",
        *dump_step_dir, manifest));
    LITERT_RETURN_IF_ERROR(DumpTensorSubset(
        decode_output_buffers, *output_kv_cache_buffers_, "kv_output",
        *dump_step_dir, manifest));
    dumped_first_decode_tensors_ = true;
  }

  if (!gpu_optimized_single_buffer_cache_) {
    std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  }
  return absl::OkStatus();
}

int LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecodeStatic(
    void* arg) {
  auto self = static_cast<LlmLiteRtCompiledModelExecutorBase*>(arg);
  // Run decode with default output_logits.
  auto status = self->BindTensorsAndRunDecode(/*output_logits=*/nullptr);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to bind tensors and run decode: " << status;
  }
  return status.raw_code();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrepareFirstDecode() {
  const bool profile_scoring = ShouldProfileScoring();
  const absl::Time total_start_time = profile_scoring ? absl::Now() : absl::Time();
  const int trace_step = llm_context_->runtime_state().current_step;
  TracePhaseBegin(trace_step, kTracePrepareFirstDecode);
  if (llm_context_->runtime_state().ran_decode && !force_prepare_needed_) {
    if (profile_scoring) {
      ABSL_LOG(INFO) << "PrepareFirstDecode fast_path_ms="
                     << absl::ToDoubleMilliseconds(absl::Now() -
                                                   total_start_time);
    }
    TracePhaseEnd(trace_step, kTracePrepareFirstDecode);
    return absl::OkStatus();
  }
  force_prepare_needed_ = false;
  // Mark that we have run decode at least once.
  llm_context_->runtime_state().ran_decode = true;

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  if (output_heads <= 1) {
    if (profile_scoring) {
      ABSL_LOG(INFO) << "PrepareFirstDecode no_copy_path_ms="
                     << absl::ToDoubleMilliseconds(absl::Now() -
                                                   total_start_time)
                     << " output_heads=" << output_heads;
    }
    TracePhaseEnd(trace_step, kTracePrepareFirstDecode);
    return absl::OkStatus();
  }

  if (output_heads > 1) {
    TracePhaseBegin(trace_step, kTracePrepareFirstDecodeBroadcast);
    const absl::Time broadcast_start_time =
        profile_scoring ? absl::Now() : absl::Time();
    LITERT_RETURN_IF_ERROR(llm_context_->processed_context()
                               .processed_tokens()
                               .BroadcastTokenCandidates(output_heads));
    if (profile_scoring) {
      ABSL_LOG(INFO) << "PrepareFirstDecode broadcast_ms="
                     << absl::ToDoubleMilliseconds(absl::Now() -
                                                   broadcast_start_time)
                     << " output_heads=" << output_heads;
    }
    TracePhaseEnd(trace_step, kTracePrepareFirstDecodeBroadcast);
  }

  const bool has_dedicated_decode_kv_cache_buffers =
      decode_kv_cache_buffers_1_.has_value() &&
      decode_kv_cache_buffers_2_.has_value();
  LITERT_RETURN_IF_ERROR(has_dedicated_decode_kv_cache_buffers);
  // Broadcast the prefill kv cache buffers to the decode kv cache buffers.
  // This is only needed when decode batch size > 1.
  const absl::Time copy_start_time =
      profile_scoring ? absl::Now() : absl::Time();
  TracePhaseBegin(trace_step, kTracePrepareFirstDecodeCopyKv);
  LITERT_RETURN_IF_ERROR(CopyKvCacheBuffers(
      output_heads, /*src_index_to_copy_on_prefill=*/-1,
      *input_kv_cache_buffers_, *decode_kv_cache_buffers_1_));
  if (profile_scoring) {
    ABSL_LOG(INFO) << "PrepareFirstDecode copy_kv_ms="
                   << absl::ToDoubleMilliseconds(absl::Now() -
                                                 copy_start_time)
                   << " output_heads=" << output_heads;
  }
  TracePhaseEnd(trace_step, kTracePrepareFirstDecodeCopyKv);
  input_kv_cache_buffers_ = &decode_kv_cache_buffers_1_.value();
  output_kv_cache_buffers_ = &decode_kv_cache_buffers_2_.value();

  if (profile_scoring) {
    ABSL_LOG(INFO) << "PrepareFirstDecode total_ms="
                   << absl::ToDoubleMilliseconds(absl::Now() -
                                                 total_start_time)
                   << " output_heads=" << output_heads;
  }
  TracePhaseEnd(trace_step, kTracePrepareFirstDecode);

  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorDecodeParams& decode_params) {

  std::vector<std::vector<int>> output_tokens_vector;
  if (mtp_drafter_ == nullptr) {
    ASSIGN_OR_RETURN(auto decoded_logits,
                     DecodeLogits(ExecutorInputs(), decode_params));
    std::optional<TensorBuffer> output_tokens;
    {
      LITERT_ASSIGN_OR_RETURN(auto decoded_logits_type,
                              decoded_logits.TensorType());
      auto dimensions = decoded_logits_type.Layout().Dimensions();
      // Shape of decoded_logits is [batch_size, Token_length, vocab_size].
      RET_CHECK_EQ(dimensions.size(), 3);
      LITERT_ASSIGN_OR_RETURN(
          output_tokens,
          CreateTensorBuffer<int>({dimensions[0], dimensions[1]}));
    }
    RETURN_IF_ERROR(SampleLogits(decoded_logits, *output_tokens));
    LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                            CopyFromTensorBuffer2D<int>(*output_tokens));
  } else {
    // MTP keeps an internal state of the last time it was called and will
    // use those projected activations to kick off the next draft steps. As
    // such, we need to do a single decode step on the first decode call after
    // prefill and provide the projected activations to the MTP drafted only
    // once.
    bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
    if (last_run_is_decode) {
      ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(ExecutorInputs()));
      RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
      // Output: [Batch, drafted and verified tokens]
      LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                              mtp_drafter_->Draft(step_and_token.step,
                                                  step_and_token.token[0]->id(),
                                                  /*activations=*/std::nullopt,
                                                  *input_kv_cache_buffers_,
                                                  *output_kv_cache_buffers_));
      RET_CHECK_EQ(output_tokens_vector.size(), 1);
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
    } else {
      int token_id = -1;
      {
        ASSIGN_OR_RETURN(auto decoded_logits,
                         DecodeLogits(ExecutorInputs(), decode_params));
        LITERT_ASSIGN_OR_RETURN(auto decoded_logits_type,
                                decoded_logits.TensorType());
        auto dimensions = decoded_logits_type.Layout().Dimensions();
        // Shape of decoded_logits is [batch_size, Token_length, vocab_size].
        RET_CHECK_EQ(dimensions.size(), 3);
        LITERT_ASSIGN_OR_RETURN(
            auto output_tokens,
            CreateTensorBuffer<int>({dimensions[0], dimensions[1]}));
        RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens));
        LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                                CopyFromTensorBuffer2D<int>(output_tokens));
        RET_CHECK_EQ(output_tokens_vector.size(), 1);
        RET_CHECK_EQ(output_tokens_vector[0].size(), 1);
        token_id = output_tokens_vector[0][0];
      }

      RET_CHECK(decode_output_buffers_.contains("activations"));
      LITERT_ASSIGN_OR_RETURN(
          auto activations, decode_output_buffers_["activations"].Duplicate());
      // Note: Position remains the same as the prefill step. However,
      // current_step is incremented in DecodeLogits and as such needs to be
      // decremented.
      LITERT_ASSIGN_OR_RETURN(
          output_tokens_vector,
          mtp_drafter_->Draft(llm_context_->runtime_state().current_step - 1,
                              token_id, std::move(activations),
                              *input_kv_cache_buffers_,
                              *output_kv_cache_buffers_));
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
      output_tokens_vector[0].insert(output_tokens_vector[0].begin(), token_id);
    }
  }

  // Check for any invalid token ids and set them to zero, if any.
  bool has_invalid_output_token = false;
  for (int batch = 0; batch < output_tokens_vector.size(); ++batch) {
    for (int token_idx = 0; token_idx < output_tokens_vector[batch].size();
         ++token_idx) {
      if (output_tokens_vector[batch][token_idx] < 0) {
        has_invalid_output_token = true;
        output_tokens_vector[batch][token_idx] = 0;
      }
    }
  }
  if (has_invalid_output_token) {
    ABSL_LOG(WARNING) << "Invalid decode and sample result. The sampled token "
                         "is casted to 0 to avoid crash.";
  }

  // Update context with the assumption that there is one output per head.
  // We must change this when doing drafter based decoding.
  std::vector<int> processed_tokens;
  std::vector<std::shared_ptr<TokenData>> pending_tokens;
  for (auto& output_head_tokens : output_tokens_vector) {
    for (int i = 0; i < output_head_tokens.size(); ++i) {
      // Last token is reserved as pending input token.
      if (i == output_head_tokens.size() - 1) {
        pending_tokens.push_back(
            std::make_shared<TokenData>(output_head_tokens[i]));
      } else {
        processed_tokens.push_back(output_head_tokens[i]);
      }
    }
  }
  if (!processed_tokens.empty()) {
    llm_context_->processed_context().processed_tokens().AddProcessedTokens(
        processed_tokens);
  }
  RETURN_IF_ERROR(
      llm_context_->processed_context().processed_tokens().AddPendingInputToken(
          pending_tokens));

  return output_tokens_vector;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorInputs& inputs, TensorBuffer& output_logits) {
  RETURN_IF_ERROR(PrepareFirstDecode());
  ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<TensorBuffer> LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {
  const bool profile_scoring = ShouldProfileScoring();
  const absl::Time total_start_time = profile_scoring ? absl::Now() : absl::Time();
  LITERT_ASSIGN_OR_RETURN(
      auto output_logits,
      decode_output_buffers_[signatures_.output_logits].Duplicate());

  bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
  const absl::Time prepare_start_time =
      profile_scoring ? absl::Now() : absl::Time();
  RETURN_IF_ERROR(PrepareFirstDecode());
  const absl::Time prepare_end_time =
      profile_scoring ? absl::Now() : absl::Time();
  const absl::Time token_start_time =
      profile_scoring ? absl::Now() : absl::Time();
  ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  const absl::Time token_end_time =
      profile_scoring ? absl::Now() : absl::Time();
  const absl::Time decode_start_time =
      profile_scoring ? absl::Now() : absl::Time();
  RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  const absl::Time decode_end_time =
      profile_scoring ? absl::Now() : absl::Time();
  const absl::Time consume_start_time =
      profile_scoring ? absl::Now() : absl::Time();
  RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
  const absl::Time consume_end_time =
      profile_scoring ? absl::Now() : absl::Time();

  if (decode_params.HasConstraintDecoder() && !step_and_token.token.empty()) {
    int output_heads = 1;
    if (llm_context_->runtime_config().output_heads.has_value()) {
      output_heads = llm_context_->runtime_config().output_heads.value();
    }

    RET_CHECK_EQ(step_and_token.token.size(), output_heads);
    std::vector<int> current_token_ids;
    current_token_ids.reserve(output_heads);
    for (const auto& token : step_and_token.token) {
      current_token_ids.push_back(token->id());
    }
    // Update constraint state only with decode ids.
    if (last_run_is_decode) {
      RETURN_IF_ERROR(
          decode_params.GetConstraintDecoder()->UpdateConstraintState(
              absl::MakeSpan(current_token_ids)));
    }

    LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_type,
                            output_logits.BufferType());
    // If the output logits are already on the host memory, use the buffer
    // directly.
    if (output_logits_buffer_type == TensorBufferType::kHostMemory) {
      // Mask logits based on the current constraint state.
      RETURN_IF_ERROR(
          decode_params.GetConstraintDecoder()->MaskLogits(output_logits));
    } else {
      // For GPU, we always copy the logits to CPU and mask them, then write
      // them back to GPU.
      LITERT_ASSIGN_OR_RETURN(RankedTensorType logits_tensor_type,
                              output_logits.TensorType());
      if (logits_tensor_type.ElementType() == ElementType::Float32) {
        // Copy the logits from the tensor buffer to a vector.
        LITERT_ASSIGN_OR_RETURN(auto logits_vector,
                                CopyFromTensorBuffer<float>(output_logits));
        // Mask logits based on the current constraint state.
        RETURN_IF_ERROR(decode_params.GetConstraintDecoder()->MaskLogits(
            absl::MakeSpan(logits_vector.data(), logits_vector.size()),
            logits_tensor_type.Layout().Dimensions()));
        // Write the masked logits back to the tensor buffer.
        output_logits.Write(
            absl::MakeConstSpan(logits_vector.data(), logits_vector.size()));
      } else if (logits_tensor_type.ElementType() ==
                 litert::ElementType::Float16) {
        // Copy the logits from the tensor buffer to a vector.
        LITERT_ASSIGN_OR_RETURN(
            auto logits_vector,
            CopyFromTensorBuffer<tflite::half>(output_logits));

        // Mask logits based on the current constraint state.
        RETURN_IF_ERROR(decode_params.GetConstraintDecoder()->MaskLogits(
            absl::MakeSpan(logits_vector.data(), logits_vector.size()),
            logits_tensor_type.Layout().Dimensions()));
        // Write the masked logits back to the tensor buffer.
        output_logits.Write(
            absl::MakeConstSpan(logits_vector.data(), logits_vector.size()));
      } else {
        return absl::InvalidArgumentError(
            "Output logits are not in float32 or float16 type.");
      }
    }
  }

  ++llm_context_->runtime_state().current_step;

  const auto& settings = executor_settings_.GetAdvancedSettings();
  if (settings && settings->num_logits_to_print_after_decode > 0) {
    if (absl::Status status =
            LogTensor(output_logits, settings->num_logits_to_print_after_decode,
                      "Logits");
        !status.ok()) {
      ABSL_LOG(WARNING) << "Failed to log logits tensor: " << status;
    }
  }
  if (profile_scoring) {
    ABSL_LOG(INFO)
        << "DecodeLogits timings: prepare_ms="
        << absl::ToDoubleMilliseconds(prepare_end_time - prepare_start_time)
        << " get_token_ms="
        << absl::ToDoubleMilliseconds(token_end_time - token_start_time)
        << " decode_internal_ms="
        << absl::ToDoubleMilliseconds(decode_end_time - decode_start_time)
        << " consume_token_ms="
        << absl::ToDoubleMilliseconds(consume_end_time - consume_start_time)
        << " total_ms="
        << absl::ToDoubleMilliseconds(absl::Now() - total_start_time)
        << " last_run_is_decode=" << last_run_is_decode
        << " token_count=" << step_and_token.token.size();
  }
  return output_logits;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::InitializeSampler(
    std::optional<ActivationDataType> logits_data_type) {
  if (sampler_ != nullptr) {
    return absl::OkStatus();
  }

  // Use the provided activation data type if available, otherwise fallback to
  // the member variable.
  auto data_type = logits_data_type.value_or(logits_data_type_);

  ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(executor_settings_));
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  ASSIGN_OR_RETURN(
      sampler_,
      CreateSampler(sampler_backend, output_heads, std::move(sampler_params),
                    env_.Get(), /*sequence_size=*/1, vocab_size, data_type));

  // If the sampler can handle input, prepare the input tensors for it.
  sampler_handles_input_ =
      (!executor_settings_.GetAdvancedSettings().has_value() ||
       executor_settings_.GetAdvancedSettings()->sampler_handles_input) &&
      sampler_->CanHandleInput() && !signatures_.input_tokens.empty();
  if (sampler_handles_input_) {
    ABSL_LOG(INFO) << "Sampler will handle decode input tensors.";
    if (!decode_prev_input_pos_) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_input_pos_,
          compiled_model_.CreateInputBuffer(kDecodeSignatureRunner,
                                            signatures_.input_positions));
    }
    if (!decode_prev_mask_ && signatures_.input_attn_mask.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_mask_,
          compiled_model_.CreateInputBuffer(kDecodeSignatureRunner,
                                            *signatures_.input_attn_mask));
    }
    // Set, then reset the input handling to get the underlying model ready, but
    // not to bind the input tensors.
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/false));
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SwapSamplerInputTensors() {
  bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
  // Move the input_pos and mask to previous ones.
  std::swap(decode_prev_input_pos_,
            decode_input_buffers_[signatures_.input_positions]);
  if (has_input_attn_mask) {
    std::swap(decode_prev_mask_,
              decode_input_buffers_[*signatures_.input_attn_mask]);
  }
  return SetSamplerInputHandling(/*reset=*/false);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetSamplerInputHandling(
    bool reset) {
  if (reset) {
    return sampler_->SetInputTensorsAndInferenceFunc(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
  return sampler_->SetInputTensorsAndInferenceFunc(
      &decode_input_buffers_[signatures_.input_tokens], &decode_prev_input_pos_,
      &decode_input_buffers_[signatures_.input_positions],
      has_input_attn_mask ? &decode_prev_mask_ : nullptr,
      has_input_attn_mask ? &decode_input_buffers_[*signatures_.input_attn_mask]
                          : nullptr,
      BindTensorsAndRunDecodeStatic, this);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SampleLogits(
    const TensorBuffer& logits, TensorBuffer& ids_tensor) {
  if (sampler_ == nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
    ActivationDataType logits_data_type;
    if (logits_tensor_type.ElementType() == ElementType::Float16) {
      logits_data_type = ActivationDataType::FLOAT16;
    } else if (logits_tensor_type.ElementType() == ElementType::Float32) {
      logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported logits data type for sampler: ",
                       static_cast<int>(logits_tensor_type.ElementType())));
    }

    RETURN_IF_ERROR(InitializeSampler(logits_data_type));
  }

  if (sampler_handles_input_) {
    RETURN_IF_ERROR(SwapSamplerInputTensors());
  }

  const int step = llm_context_->runtime_state().current_step - 1;
  TracePhaseBegin(step, kTraceDecodeSampling);
  BenchmarkMark(benchmark_info_, "llm_decode_sampling");
  RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  BenchmarkMark(benchmark_info_, "llm_decode_sampling");
  TracePhaseEnd(step, kTraceDecodeSampling);
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::UpdateExecutorSettings(
    const LlmExecutorSettings& executor_settings) {
  executor_settings_ = executor_settings;
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::UpdateRuntimeConfig(
    const RuntimeConfig& runtime_config) {
  const int old_output_heads = GetOutputHeads(llm_context_->runtime_config());
  const int new_output_heads = GetOutputHeads(runtime_config);
  if (new_output_heads <= 0) {
    return absl::InvalidArgumentError(
        "RuntimeConfig.output_heads must be positive.");
  }
  if (old_output_heads > 1 && new_output_heads > 1 &&
      old_output_heads != new_output_heads) {
    return absl::UnimplementedError(
        absl::StrCat("Changing output_heads from ", old_output_heads, " to ",
                     new_output_heads, " is not supported."));
  }

  if (old_output_heads > 1 && new_output_heads == 1) {
    constexpr int kTokenIndexToReduce = 0;
    RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(kTokenIndexToReduce));
  }

  llm_context_->runtime_config() = runtime_config;
  if (old_output_heads != new_output_heads) {
    sampler_.reset();
    force_prepare_needed_ = new_output_heads > 1;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetCurrentStep(int new_step) {
  ASSIGN_OR_RETURN(auto old_step, GetCurrentStep());
  if (old_step == new_step) {
    return absl::OkStatus();
  }

  int max_step = old_step;
  RET_CHECK_LE(new_step, max_step).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: " << max_step;
  RET_CHECK_GE(new_step, 0).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be negative.";
  if (new_step == max_step) {
    llm_context_->runtime_state().current_step = new_step;
    return absl::OkStatus();
  }
  RET_CHECK_LE(new_step, max_step).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: " << max_step;
  if (new_step < 0) {
    // Current step is negative after rolling back. This can only happen when
    // the user wants to set the step to 0 while there is a pending input token.
    // Thus we can roll back executor state to step 0.
    return Reset();
  }
  llm_context_->runtime_state().current_step = new_step;

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Reset() {
  llm_context_->runtime_state().current_step = 0;
  llm_context_->runtime_state().ran_decode = false;
  llm_context_->processed_context().processed_tokens() = ProcessedTokens();
  input_kv_cache_buffers_ = &kv_cache_buffers_1_;
  output_kv_cache_buffers_ = &kv_cache_buffers_2_;
  force_prepare_needed_ = false;
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutorBase::GetVocabSize() {
  if (!decode_output_buffers_.contains(signatures_.output_logits)) {
    return absl::NotFoundError("Output logits info not found.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      decode_output_buffers_[signatures_.output_logits].TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  return logits_tensor_type.Layout().Dimensions()[2];
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorStatic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorStatic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  // For now, we reduce the input and processed tokens for prefill only with
  // the first input and processed tokens. This should be updated if user select
  // the decode output candidate.
  constexpr int kTokenIndexToReduce = 0;
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(kTokenIndexToReduce));

  LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_buffer->TensorType());
  // Accept batch size 1 or output_heads though prefill handles only the
  // first batch element.
  int32_t input_batch_size = tensor_type.Layout().Dimensions()[0];
  if (input_batch_size != 1) {
    RET_CHECK_EQ(input_batch_size, output_heads);
  }
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";

  if (embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }

  LITERT_ASSIGN_OR_RETURN(auto ids,
                          ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));
  // Reduce the input ids only with one user selected.
  auto input_length = ids.size() / input_batch_size;
  ids = ids.subspan(kTokenIndexToReduce * input_length, input_length);
  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (int i = 0; i < work_groups.size(); ++i) {
    const auto& prefill_signature = work_groups[i].first;
    int prefill_length = work_groups[i].second;
    // Keep track of the signatures that have already had their buffers
    // created only create them once.
    if (!prefill_input_buffers_.contains(prefill_signature)) {
      prefill_input_buffers_[prefill_signature] = {};
      RETURN_IF_ERROR(CreatePrefillInputBuffers(
          prefill_signature, prefill_length, prefill_length,
          prefill_input_buffers_[prefill_signature]));
    }
    // TODO(b/494284915): Switch to use async prefill for Metal backend.
    if (!do_prefill_sync_.has_value()) {
      do_prefill_sync_ = std::any_of(
          prefill_input_buffers_[prefill_signature].begin(),
          prefill_input_buffers_[prefill_signature].end(),
          [](const auto& pair) { return pair.second.IsMetalMemory(); });
    }
    bool async = !*do_prefill_sync_ &&
                 (i < work_groups.size() - 1 || !params.GetWaitForCompletion());
    RETURN_IF_ERROR(PrefillInternal(
        prefill_signature, prefill_input_buffers_[prefill_signature],
        ids.subspan(/*pos=*/0, prefill_length), async));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(embedding_lookup_->CleanupMultiModalEmbeddings());
  }

  return absl::OkStatus();
}

// static
// Creates a LlmLiteRtCompiledModelExecutorStatic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic>>
LlmLiteRtCompiledModelExecutorStatic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  std::string cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  // TODO(b/433590109): Some GPUs do not support FP16, so we need to check the
  // capabilities of the GPU and set the activation data type accordingly.
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type = executor_settings.GetActivationDataType().value();
  }
  const Backend backend = executor_settings.GetBackend();
  bool use_fp16_precision =
      activation_data_type == ActivationDataType::FLOAT16 &&
      backend == Backend::GPU;

  if (!litert_model || !*litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
  }

  absl::string_view prefill_signature_key = "";
  for (int i = 0; i < litert_model->GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, litert_model->GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto prefill_signature,
                          litert_model->FindSignature(prefill_signature_key));
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetKVCacheRootNames(
      prefill_signature.InputNames(), prefill_signature.OutputNames(),
      kv_cache_k_root_name, kv_cache_v_root_name));
  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  bool gpu_optimized_single_buffer_cache =
      backend == Backend::GPU && signatures.input_int32_param.has_value();

  LITERT_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, activation_data_type,
                               &signatures));

  auto section_offset =
      resources.GetWeightsSectionOffset(ModelType::kTfLitePrefillDecode);
  if (section_offset.ok()) {
    if (backend != Backend::GPU) {
      return absl::InvalidArgumentError(
          "Weights section offset is only "
          "supported for GPU backend.");
    }
    Options::ScopedWeightSectionMap section_map;
    section_map["tflite_weights"] = {
        section_offset.value().first,
        section_offset.value().second - section_offset.value().first};
    ABSL_LOG(INFO) << "section_map: " << section_map["tflite_weights"].offset
                   << " " << section_map["tflite_weights"].length;
    LITERT_ASSIGN_OR_RETURN(auto scoped_file, resources.GetScopedFile());
    compilation_options.SetExternalWeightScopedFile(scoped_file.get(),
                                                    section_map);
  };

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(lrt_env, litert_model->Get(), compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_kv_cache_buffers;

  bool clear_kv_cache_before_prefill =
      !executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->clear_kv_cache_before_prefill;
  for (auto input_name : prefill_signature.InputNames()) {
    // Skip creating buffers for the input tokens, positions and attn mask. Move
    // into prefill function to create them based on the ids size.
    if (!IsKVCacheTensor(input_name) || gpu_optimized_single_buffer_cache) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model.CreateInputBuffer(prefill_signature_key, input_name));
    MaybeLogCompiledModelBufferCreation("prefill_setup", prefill_signature_key,
                                        input_name,
                                        /*is_input=*/true, input_buffer);
    if (clear_kv_cache_before_prefill) {
      LITERT_RETURN_IF_ERROR(input_buffer.Clear());
    }
    if (backend == Backend::CPU) {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer, input_buffer.Duplicate());
      output_kv_cache_buffers[input_name] = std::move(output_buffer);
    }
    input_kv_cache_buffers[input_name] = std::move(input_buffer);
  }
  for (auto output_name : prefill_signature.OutputNames()) {
    if (IsKVCacheTensor(output_name)) {
      if (backend == Backend::GPU) {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(
                                    prefill_signature_key, output_name));
        MaybeLogCompiledModelBufferCreation("prefill_setup",
                                            prefill_signature_key, output_name,
                                            /*is_input=*/false, output_buffer);
        if (clear_kv_cache_before_prefill &&
            gpu_optimized_single_buffer_cache) {
          LITERT_RETURN_IF_ERROR(output_buffer.Clear());
        }
        output_kv_cache_buffers[output_name] = std::move(output_buffer);
      }
      // For CPU, we will use single buffer for kv cache input and output to
      // improve performance and memory usage.
    } else {
      // TODO b/444063139 - Support non-kv_cache tensors as prefill outputs.
      // This should be done once we have a model that has non-kv_cache tensors
      // as prefill outputs. It should be done in the same place as the prefill
      // inputs are created.
      return absl::UnimplementedError(absl::StrCat(
          "Failed to create prefill output buffer for '", output_name,
          "'. Only kv_cache tensors are supported as outputs to "
          "prefill at the moment."));
    }
  }

  for (auto input_name : decode_signature.InputNames()) {
    if (IsLoRAInputName(input_name)) {
      // We let LoraManager handle LoRA inputs.
      continue;
    }
    if (IsKVCacheTensor(input_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model.CreateInputBuffer(kDecodeSignatureRunner, input_name));
    decode_input_buffers[input_name] = std::move(input_buffer);
  }
  auto output_names = decode_signature.OutputNames();
  for (int i = 0; i < output_names.size(); ++i) {
    auto output_name = output_names[i];
    if (IsKVCacheTensor(output_name)) {
      continue;
    }
    // If we are using the GPU sampler and the model is compiled with FP16
    // precision, we force the output logits to be FP16 as the
    // GPU sampler supports FP16 inputs.
    // If we use CPU sampler or the model is executed with FP32 / mixed
    // precision, we will keep the logits in FP32
    auto sampler_backend = GetSamplerBackend(executor_settings);

    if (output_name == signatures.output_logits && use_fp16_precision &&
        sampler_backend.ok() && *sampler_backend == Backend::GPU) {
      LITERT_ASSIGN_OR_RETURN(
          size_t signature_index,
          compiled_model.GetSignatureIndex(kDecodeSignatureRunner));
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateFP16OutputBuffer(lrt_env, compiled_model, signature_index,
                                 output_name, i));
      decode_output_buffers[output_name] = std::move(output_buffer);
    } else {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                              compiled_model.CreateOutputBuffer(
                                  kDecodeSignatureRunner, output_name));

      decode_output_buffers[output_name] = std::move(output_buffer);
    }
  }

  if (signatures.input_attn_mask.has_value()) {
    const auto mask_it = decode_input_buffers.find(*signatures.input_attn_mask);
    if (mask_it != decode_input_buffers.end()) {
      ASSIGN_OR_RETURN(const int decode_mask_capacity,
                       GetAttentionMaskCapacity(mask_it->second));
      const uint32_t configured_max_num_tokens =
          executor_settings.GetMaxNumTokens();
      if (configured_max_num_tokens == 0) {
        ABSL_LOG(INFO)
            << "Setting max_num_tokens to static decode attention mask "
               "capacity "
            << decode_mask_capacity
            << " because no runtime token limit was configured.";
        executor_settings.SetMaxNumTokens(decode_mask_capacity);
      } else {
        RET_CHECK_LE(configured_max_num_tokens,
                     static_cast<uint32_t>(decode_mask_capacity))
                .SetCode(absl::StatusCode::kInvalidArgument)
            << "Configured max_num_tokens=" << configured_max_num_tokens
            << " exceeds static decode attention mask capacity="
            << decode_mask_capacity;
      }
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto output_logits_buffer,
      decode_output_buffers[signatures.output_logits].Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_tensor_type,
                          output_logits_buffer.TensorType());
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];

  std::optional<absl::flat_hash_map<absl::string_view, TensorBuffer>>
      decode_input_kv_cache_buffers;
  std::optional<absl::flat_hash_map<absl::string_view, TensorBuffer>>
      decode_output_kv_cache_buffers;
  const bool allocate_dedicated_decode_kv_cache_buffers =
      batch_size > 1 ||
      (backend == Backend::GPU && !gpu_optimized_single_buffer_cache);
  if (allocate_dedicated_decode_kv_cache_buffers) {
    if (batch_size > 1) {
      ABSL_LOG(INFO) << "Decode batch size is larger than 1. Allocate decode "
                     << "only KV cache buffers.";
    } else {
      ABSL_LOG(INFO)
          << "GPU backend detected. Allocate dedicated decode KV cache "
             "buffers to avoid reusing prefill buffers across signatures.";
    }
    decode_input_kv_cache_buffers =
        absl::flat_hash_map<absl::string_view, TensorBuffer>();
    decode_output_kv_cache_buffers =
        absl::flat_hash_map<absl::string_view, TensorBuffer>();
    for (auto input_name : decode_signature.InputNames()) {
      if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
          absl::StartsWith(input_name, kv_cache_v_root_name)) {
        LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                                compiled_model.CreateInputBuffer(
                                    kDecodeSignatureRunner, input_name));
        MaybeLogCompiledModelBufferCreation("decode_setup",
                                            kDecodeSignatureRunner, input_name,
                                            /*is_input=*/true, input_buffer);
        if (clear_kv_cache_before_prefill) {
          LITERT_RETURN_IF_ERROR(input_buffer.Clear());
        }
        (*decode_input_kv_cache_buffers)[input_name] = std::move(input_buffer);
      }
    }
    for (auto output_name : decode_signature.OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_v_root_name)) {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(
                                    kDecodeSignatureRunner, output_name));
        MaybeLogCompiledModelBufferCreation("decode_setup",
                                            kDecodeSignatureRunner, output_name,
                                            /*is_input=*/false, output_buffer);
        (*decode_output_kv_cache_buffers)[output_name] =
            std::move(output_buffer);
      }
    }
  }

  ASSIGN_OR_RETURN(auto prefill_runner_set,
                   GetPrefillRunnerSetFromModel(
                       *litert_model, kPrefillSignatureRunner,
                       /*input_positions_name=*/signatures.input_positions));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  RETURN_IF_ERROR(InitializeEmbeddingLookups(
      lrt_env, resources, embedding_lookup, per_layer_embedding_lookup));
  std::unique_ptr<LlmLiteRtMtpDrafter> mtp_drafter;
  {
    const auto& advanced_settings = executor_settings.GetAdvancedSettings();
    if (advanced_settings.has_value() &&
        advanced_settings->enable_speculative_decoding) {
      RET_CHECK_NE(embedding_lookup, nullptr);
      RET_CHECK_NE(per_layer_embedding_lookup, nullptr);
      LITERT_ASSIGN_OR_RETURN(
          auto base_compiled_model,
          CompiledModel::Create(lrt_env, litert_model->Get(),
                                compilation_options));
      ASSIGN_OR_RETURN(mtp_drafter,
                       LlmLiteRtMtpDrafter::Create(
                           lrt_env, resources, executor_settings,
                           std::move(base_compiled_model), *embedding_lookup,
                           *per_layer_embedding_lookup));
    }
  }

  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutorStatic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(input_kv_cache_buffers),
      std::move(output_kv_cache_buffers),
      std::move(decode_input_kv_cache_buffers),
      std::move(decode_output_kv_cache_buffers), std::move(prefill_runner_set),
      signatures, batch_size, std::move(cache_path),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      use_fp16_precision, activation_data_type, std::move(mtp_drafter)));
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorDynamic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorDynamic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  // Only accept batch size 1 for now.
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(0));

  LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_buffer->TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN(
      absl::Span<int> ids, ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));

  if (prefill_chunk_size_ <= 0) {
    return PrefillInternal(ids, params);
  }

  while (!ids.empty()) {
    int chunk_size =
        std::min(static_cast<int>(ids.size()), prefill_chunk_size_);
    absl::Span<int> chunk_ids = ids.first(chunk_size);
    ids = ids.subspan(chunk_size);
    RETURN_IF_ERROR(PrefillInternal(chunk_ids, params));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::PrefillInternal(
    absl::Span<int> ids, const ExecutorPrefillParams& params) {
  RETURN_IF_ERROR(RollBackProcessedTokens());
  // Check if have a pending input token. Note that 'internal_start_step' is
  // always equal to the number of processed tokens plus 1.
  ProcessedTokens::StepAndToken step_and_token =
      llm_context_->processed_context()
          .processed_tokens()
          .GetNextUnprocessedToken();
  bool has_pending_input_token = !step_and_token.token.empty();
  int prefill_length = has_pending_input_token ? ids.size() : ids.size() - 1;
  // If there is no pending input token and no input token to prefill, we can
  // return early by storing the token as a pending input token.
  if (!has_pending_input_token && prefill_length == 0) {
    RETURN_IF_ERROR(
        llm_context_->processed_context()
            .processed_tokens()
            .AddPendingInputToken({std::make_shared<TokenData>(ids[0])}));
    return absl::OkStatus();
  }

  int kv_length = 0;
  if (kv_cache_buffers_1_.empty()) {
    kv_length = prefill_length;
    // First time prefilling, allocate KV cache buffers.
    bool clear_kv_cache_before_prefill =
        !executor_settings_.GetAdvancedSettings() ||
        executor_settings_.GetAdvancedSettings()->clear_kv_cache_before_prefill;
    for (const auto& k_cache_input_name : key_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                          k_cache_input_name, prefill_length));
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model_.CreateInputBuffer("prefill", k_cache_input_name));
      if (clear_kv_cache_before_prefill) {
        LITERT_RETURN_IF_ERROR(input_buffer.Clear());
      }
      kv_cache_buffers_1_[k_cache_input_name] = std::move(input_buffer);
    }
    for (const auto& v_cache_input_name : value_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                          v_cache_input_name, prefill_length));
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model_.CreateInputBuffer("prefill", v_cache_input_name));
      if (clear_kv_cache_before_prefill) {
        LITERT_RETURN_IF_ERROR(input_buffer.Clear());
      }
      kv_cache_buffers_1_[v_cache_input_name] = std::move(input_buffer);
    }
  } else {
    {
      RET_CHECK(!resizable_key_cache_input_names_.empty());
      RET_CHECK(!kv_cache_buffers_1_.empty());
      const TensorBuffer& key_buffer =
          kv_cache_buffers_1_[resizable_key_cache_input_names_[0]];
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType& key_buffer_tensor_type,
                              key_buffer.TensorType());
      kv_length = key_buffer_tensor_type.Layout().Dimensions()
          [key_dynamic_dim_indices_.at(resizable_key_cache_input_names_[0])];
    }

    int free_kv_entries = kv_length - step_and_token.step;
    if (prefill_length > free_kv_entries) {
      int new_kv_seq_len = kv_length + prefill_length;
      int entries_to_add = new_kv_seq_len - kv_length;
      for (const auto& k_cache_input_name : resizable_key_cache_input_names_) {
        RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                            k_cache_input_name,
                                            new_kv_seq_len));
        ASSIGN_OR_RETURN(kv_cache_buffers_1_[k_cache_input_name],
                         ResizeKVCacheTensorBuffer(
                             env_, kv_cache_buffers_1_[k_cache_input_name],
                             key_dynamic_dim_indices_.at(k_cache_input_name),
                             entries_to_add));
      }
      for (const auto& v_cache_input_name :
           resizable_value_cache_input_names_) {
        RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                            v_cache_input_name,
                                            new_kv_seq_len));
        ASSIGN_OR_RETURN(kv_cache_buffers_1_[v_cache_input_name],
                         ResizeKVCacheTensorBuffer(
                             env_, kv_cache_buffers_1_[v_cache_input_name],
                             value_dynamic_dim_indices_.at(v_cache_input_name),
                             entries_to_add));
      }
      kv_length = new_kv_seq_len;
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers;
  RETURN_IF_ERROR(CreatePrefillInputBuffers("prefill", prefill_length,
                                            kv_length, prefill_input_buffers));

  input_kv_cache_buffers_ = &kv_cache_buffers_1_;
  output_kv_cache_buffers_ = &kv_cache_buffers_1_;

  bool async = !params.GetWaitForCompletion();
  return LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
      "prefill", prefill_input_buffers, ids, async);
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  int current_kv_len = 0;
  {
    RET_CHECK(!resizable_key_cache_input_names_.empty());
    RET_CHECK(!kv_cache_buffers_1_.empty());
    const TensorBuffer& key_buffer =
        kv_cache_buffers_1_[resizable_key_cache_input_names_[0]];
    LITERT_ASSIGN_OR_RETURN(const RankedTensorType& key_buffer_tensor_type,
                            key_buffer.TensorType());
    current_kv_len = key_buffer_tensor_type.Layout().Dimensions()
        [key_dynamic_dim_indices_.at(resizable_key_cache_input_names_[0])];
  }

  if (current_kv_len <= llm_context_->runtime_state().current_step - 1) {
    int entries_to_add = kv_increament_size_;
    int new_kv_len = current_kv_len + entries_to_add;
    for (const auto& k_cache_input_name : resizable_key_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "decode",
                                          k_cache_input_name, new_kv_len));
      ASSIGN_OR_RETURN(kv_cache_buffers_1_[k_cache_input_name],
                       ResizeKVCacheTensorBuffer(
                           env_, kv_cache_buffers_1_[k_cache_input_name],
                           key_dynamic_dim_indices_.at(k_cache_input_name),
                           entries_to_add));
    }
    for (const auto& v_cache_input_name :
         resizable_value_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "decode",
                                          v_cache_input_name, new_kv_len));
      ASSIGN_OR_RETURN(kv_cache_buffers_1_[v_cache_input_name],
                       ResizeKVCacheTensorBuffer(
                           env_, kv_cache_buffers_1_[v_cache_input_name],
                           value_dynamic_dim_indices_.at(v_cache_input_name),
                           entries_to_add));
    }
    current_kv_len = new_kv_len;
  }

  RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "decode",
                                      signatures_.input_attn_mask.value(),
                                      current_kv_len));
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers_[signatures_.input_attn_mask.value()],
      compiled_model_.CreateInputBuffer("decode",
                                        signatures_.input_attn_mask.value()));

  return LlmLiteRtCompiledModelExecutorBase::DecodeInternal(token,
                                                            output_logits);
}

// static
// Creates a LlmLiteRtCompiledModelExecutorDynamic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic>>
LlmLiteRtCompiledModelExecutorDynamic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, ActivationDataType::FLOAT32,
                               /*signatures=*/std::nullopt));
  std::string weight_cache_path = executor_settings.GetCacheDir();
  const Backend backend = executor_settings.GetBackend();
  RET_CHECK_EQ(backend, Backend::CPU)
      << "LlmLiteRtCompiledModelExecutorDynamic only supports CPU backend.";
  uint32_t kv_increament_size = 0;
  int prefill_chunk_size = -1;
  {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_compilation_options,
                            compilation_options.GetCpuOptions());
    ASSIGN_OR_RETURN(const auto& cpu_config,
                     executor_settings.GetBackendConfig<CpuConfig>());
    kv_increament_size = cpu_config.kv_increment_size;
    prefill_chunk_size = cpu_config.prefill_chunk_size;
    cpu_compilation_options.SetNumThreads(cpu_config.number_of_threads);
    auto weight_cache_file = executor_settings.GetWeightCacheFile(
        ExecutorSettingsBase::kXnnpackCacheSuffix, /*check_and_clean=*/true);
    if (weight_cache_file.ok()) {
      if (std::holds_alternative<std::string>(*weight_cache_file)) {
        weight_cache_path = std::get<std::string>(*weight_cache_file);
        ABSL_LOG(INFO) << "Setting XNNPACK weight cache path: "
                       << weight_cache_path;
        cpu_compilation_options.SetXNNPackWeightCachePath(
            weight_cache_path.c_str());
      } else {
        auto scoped_cache_file =
            std::get<std::shared_ptr<ScopedFile>>(*weight_cache_file);
        ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
        ASSIGN_OR_RETURN(int fd, duplicated.Release());
        cpu_compilation_options.SetXNNPackWeightCacheFileDescriptor(fd);
      }
    }
    RET_CHECK_GT(kv_increament_size, 0)
        << "KV increment size must be greater than 0.";
    auto default_xnn_options = TfLiteXNNPackDelegateOptionsDefault();
    cpu_compilation_options.SetXNNPackFlags(
        default_xnn_options.flags |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
    LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                            compilation_options.GetRuntimeOptions());
    runtime_options.SetCompressQuantizationZeroPoints(true);
    AdvancedSettings advanced_settings;
    if (executor_settings.GetAdvancedSettings()) {
      advanced_settings = *executor_settings.GetAdvancedSettings();
    }
    if (ShouldEnableFirstDecodeProfiling(advanced_settings)) {
      LITERT_RETURN_IF_ERROR(runtime_options.SetEnableProfiling(true));
    }
    compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(lrt_env, litert_model->Get(), compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetKVCacheRootNames(
      decode_signature.InputNames(), decode_signature.OutputNames(),
      kv_cache_k_root_name, kv_cache_v_root_name));
  ASSIGN_OR_RETURN(
      CacheAdapterMetadata cache_metadata,
      GetCacheAdapterMetadata(resources, ModelType::kTfLitePrefillDecode));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;
  std::vector<std::string> resizable_key_cache_input_names;
  std::vector<std::string> resizable_value_cache_input_names;
  for (auto input_name : decode_signature.InputNames()) {
    bool is_key_cache_input =
        absl::StartsWith(input_name, kv_cache_k_root_name);
    bool is_sequence_cache_input = false;
    if (is_key_cache_input) {
      key_cache_input_names.push_back(std::string(input_name));
      ASSIGN_OR_RETURN(is_sequence_cache_input,
                       IsSequenceCacheTensorName(
                           input_name, kv_cache_k_root_name,
                           kv_cache_v_root_name, cache_metadata));
      if (is_sequence_cache_input) {
        resizable_key_cache_input_names.push_back(std::string(input_name));
      }
    }

    bool is_value_cache_input =
        absl::StartsWith(input_name, kv_cache_v_root_name);
    if (is_value_cache_input) {
      value_cache_input_names.push_back(std::string(input_name));
      ASSIGN_OR_RETURN(is_sequence_cache_input,
                       IsSequenceCacheTensorName(
                           input_name, kv_cache_k_root_name,
                           kv_cache_v_root_name, cache_metadata));
      if (is_sequence_cache_input) {
        resizable_value_cache_input_names.push_back(std::string(input_name));
      }
    }

    bool is_kv_cache_input = is_key_cache_input || is_value_cache_input;
    bool is_attn_mask_input =
        signatures.input_attn_mask.has_value() &&
        absl::StartsWith(input_name, signatures.input_attn_mask.value());
    if (!is_kv_cache_input && !is_attn_mask_input) {
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model.CreateInputBuffer(kDecodeSignatureRunner, input_name));
      decode_input_buffers[input_name] = std::move(input_buffer);
    }
  }
  for (auto output_name : decode_signature.OutputNames()) {
    if (!absl::StartsWith(output_name, kv_cache_k_root_name) &&
        !absl::StartsWith(output_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                              compiled_model.CreateOutputBuffer(
                                  kDecodeSignatureRunner, output_name));
      decode_output_buffers[output_name] = std::move(output_buffer);
    }
  }

  RET_CHECK(!resizable_key_cache_input_names.empty())
      << "Dynamic executor requires at least one sequence key cache tensor.";
  RET_CHECK_EQ(resizable_key_cache_input_names.size(),
               resizable_value_cache_input_names.size())
      << "Sequence key/value cache tensor counts must match.";
  absl::flat_hash_map<std::string, int> key_dynamic_dim_indices;
  absl::flat_hash_map<std::string, int> value_dynamic_dim_indices;
  for (const auto& input_name : resizable_key_cache_input_names) {
    ASSIGN_OR_RETURN(key_dynamic_dim_indices[input_name],
                     GetDynamicDimIndex(*litert_model, "prefill", input_name));
  }
  for (const auto& input_name : resizable_value_cache_input_names) {
    ASSIGN_OR_RETURN(value_dynamic_dim_indices[input_name],
                     GetDynamicDimIndex(*litert_model, "prefill", input_name));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto output_logits_buffer,
      decode_output_buffers[signatures.output_logits].Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_tensor_type,
                          output_logits_buffer.TensorType());
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];
  RET_CHECK_EQ(batch_size, 1) << "Only support batch size 1 for now.";
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  RETURN_IF_ERROR(InitializeEmbeddingLookups(
      lrt_env, resources, embedding_lookup, per_layer_embedding_lookup));
  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutorDynamic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), prefill_chunk_size,
      std::move(key_dynamic_dim_indices),
      std::move(value_dynamic_dim_indices), kv_increament_size,
      std::move(key_cache_input_names), std::move(value_cache_input_names),
      std::move(resizable_key_cache_input_names),
      std::move(resizable_value_cache_input_names), signatures, batch_size,
      std::move(weight_cache_path), std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup), /*use_fp16_precision=*/false,
      /*logits_data_type=*/LogitsDataType::FLOAT32));
}

}  // namespace litert::lm
