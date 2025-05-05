#include "runtime/core/session_basic.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/sampler_factory.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/pipeline.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// Default batch size for the output. This should be configurable in the
// future.
constexpr int kOutputBatchSize = 1;

}  // namespace

absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const std::vector<int>& stop_token_ids, const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info) {
  ASSIGN_OR_RETURN(auto sampler,
                   CreateSampler(Backend::CPU, kOutputBatchSize,
                                 session_config.GetSamplerParams()));
  if (benchmark_info.has_value()) {
    ABSL_LOG(INFO) << "Benchmark is enabled.";
  }
  return absl::WrapUnique(new SessionBasic(executor, tokenizer, stop_token_ids,
                                           std::move(sampler), session_config,
                                           benchmark_info));
}

absl::Status SessionBasic::PrefillInternal(absl::string_view input,
                                           bool wait_for_completion) {
  // TODO(b/397975034): Consider to utilize a prompt formatting logic in a
  // separate library/class.
  ABSL_LOG(INFO) << "RunPrefillSync: " << input;
  ASSIGN_OR_RETURN(last_prefill_token_id_,
                   Prefill(executor_, tokenizer_, input, /*bos_token_id=*/2,
                           wait_for_completion, benchmark_info_));
  ABSL_LOG(INFO) << "Prefill done";
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(absl::string_view input) {
  return PrefillInternal(input, /*wait_for_completion=*/true);
}

absl::Status SessionBasic::RunPrefillAsync(absl::string_view input) {
  return PrefillInternal(input, /*wait_for_completion=*/false);
}

absl::StatusOr<Responses> SessionBasic::RunDecode() {
  ABSL_LOG(INFO) << "RunDecodeSync";
  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(auto responses, Decode(executor_, tokenizer_,
                                            stop_token_ids_, benchmark_info_));
    return responses;
  } else {
    std::vector<int> decoded_ids(kOutputBatchSize, last_prefill_token_id_);
    auto decoded_ids_buffer =
        CopyToTensorBuffer<int>(decoded_ids, {kOutputBatchSize, 1});
    ASSIGN_OR_RETURN(auto responses, DecodeCustomSampling(
                                         executor_, tokenizer_, stop_token_ids_,
                                         /*num_output_candidates=*/1, *sampler_,
                                         *decoded_ids_buffer, benchmark_info_));
    return responses;
  }
}

absl::StatusOr<BenchmarkInfo> SessionBasic::GetBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

}  // namespace litert::lm
