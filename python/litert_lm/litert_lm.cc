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

#include <deque>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"   // IWYU pragma: keep
#include "nanobind/stl/variant.h"      // IWYU pragma: keep
#include "nanobind/stl/vector.h"       // IWYU pragma: keep
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nanobind_json/nanobind_json.hpp"  // from @nanobind_json  // IWYU pragma: keep
#include "litert/c/internal/litert_logging.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "tflite/core/c/c_api_types.h"  // from @litert
#include "tflite/logger.h"  // from @litert
#include "tflite/minimal_logging.h"  // from @litert

#define VALUE_OR_THROW(status_or)                                   \
  ([&]() {                                                          \
    auto status_or_value = (status_or);                             \
    if (!status_or_value.ok()) {                                    \
      std::stringstream ss;                                         \
      ss << __FILE__ << ":" << __LINE__ << ": " << __func__ << ": " \
         << status_or_value.status();                               \
      throw std::runtime_error(ss.str());                           \
    }                                                               \
    return std::move(status_or_value).value();                      \
  }())

#define STATUS_OR_THROW(status)                                     \
  {                                                                 \
    auto status_value = (status);                                   \
    if (!status_value.ok()) {                                       \
      std::stringstream ss;                                         \
      ss << __FILE__ << ":" << __LINE__ << ": " << __func__ << ": " \
         << status_value;                                           \
      throw std::runtime_error(ss.str());                           \
    }                                                               \
  }

namespace litert::lm {

namespace nb = nanobind;

// The maximum number of times the model can call tools in a single turn before
// an error is thrown.
constexpr int kRecurringToolCallLimit = 25;

// Helper to convert Python dict or str to JSON message.
nlohmann::json ParseJsonMessage(const nb::handle& message) {
  if (nb::isinstance<nb::dict>(message)) {
    return nb::cast<nb::dict>(message);
  }
  if (nb::isinstance<nb::str>(message)) {
    return {{"role", "user"}, {"content", nb::cast<std::string>(message)}};
  }
  throw std::runtime_error("Message must be a dict or a str.");
}

// Helper to extract C++ Backend from Python Backend enum.
Backend ParseBackend(const nb::handle& handle,
                     Backend default_val = Backend::CPU) {
  if (handle.is_none()) return default_val;
  return static_cast<Backend>(nb::cast<int>(nb::object(handle.attr("value"))));
}

// Helper to handle tool calls.
nlohmann::json HandleToolCalls(const nlohmann::json& response,
                               const nb::dict& tool_map,
                               const nb::object& tool_event_handler) {
  nlohmann::json tool_responses = nlohmann::json::array();
  for (const auto& tool_call : response["tool_calls"]) {
    if (!tool_call.contains("function")) continue;
    std::string name = tool_call["function"]["name"];
    nlohmann::json arguments = tool_call["function"]["arguments"];

    if (!tool_event_handler.is_none()) {
      bool allowed = nb::cast<bool>(
          tool_event_handler.attr("approve_tool_call")(nb::cast(tool_call)));
      if (!allowed) {
        tool_responses.push_back({
            {"type", "tool_response"},
            {"name", name},
            {"response", "Error: Tool call disallowed by user."},
        });
        continue;
      }
    }

    nlohmann::json json_result;
    if (tool_map.contains(name.c_str())) {
      nb::object tool_obj = tool_map[name.c_str()];
      nb::object py_args = nb::cast(arguments);
      try {
        nb::object py_result = tool_obj.attr("execute")(py_args);
        json_result = nb::cast<nlohmann::json>(py_result);
      } catch (const std::exception& e) {
        json_result = "Error executing tool: " + std::string(e.what());
      }
    } else {
      json_result = "Error: Tool not found.";
    }

    nlohmann::json tool_response_json = {
        {"name", name},
        {"response", json_result},
    };

    if (!tool_event_handler.is_none()) {
      nb::object py_modified_response = tool_event_handler.attr(
          "process_tool_response")(nb::cast(tool_response_json));
      tool_response_json = nb::cast<nlohmann::json>(py_modified_response);
    }

    tool_responses.push_back({
        {"type", "tool_response"},
        {"name", name},
        {"response", json_result},
    });
  }

  return {{"role", "tool"}, {"content", tool_responses}};
}

// Helper to inject Python backend attribute.
void SetBackendAttr(nb::object& py_engine, const nb::handle& backend_handle) {
  if (backend_handle.is_none()) {
    py_engine.attr("backend") =
        nb::module_::import_(
            "litert_lm.interfaces")
            .attr("Backend")
            .attr("CPU");
  } else {
    py_engine.attr("backend") = backend_handle;
  }
}

template <typename T>
std::optional<T> GetOptionalAttr(const nb::object& obj, const char* attr_name) {
  if (!nb::hasattr(obj, attr_name)) {
    return std::nullopt;
  }
  nb::object attr = obj.attr(attr_name);
  if (attr.is_none()) {
    return std::nullopt;
  }
  return nb::cast<T>(attr);
}

SessionConfig ParseSessionOptions(const nb::handle& options) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  if (options.is_none()) {
    return session_config;
  }
  nb::object options_obj = nb::borrow<nb::object>(options);
  auto apply_prompt_template_in_session = GetOptionalAttr<bool>(
      options_obj, "apply_prompt_template_in_session");
  if (apply_prompt_template_in_session.has_value()) {
    session_config.SetApplyPromptTemplateInSession(
        *apply_prompt_template_in_session);
  }
  auto max_output_tokens = GetOptionalAttr<int>(options_obj, "max_output_tokens");
  if (max_output_tokens.has_value()) {
    session_config.SetMaxOutputTokens(*max_output_tokens);
  }
  auto num_output_candidates =
      GetOptionalAttr<int>(options_obj, "num_output_candidates");
  if (num_output_candidates.has_value()) {
    session_config.SetNumOutputCandidates(*num_output_candidates);
  }
  return session_config;
}

DecodeConfig ParseDecodeOptions(const nb::handle& options) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  if (options.is_none()) {
    return decode_config;
  }
  nb::object options_obj = nb::borrow<nb::object>(options);
  auto max_output_tokens = GetOptionalAttr<int>(options_obj, "max_output_tokens");
  if (max_output_tokens.has_value()) {
    decode_config.SetMaxOutputTokens(*max_output_tokens);
  }
  return decode_config;
}

// Helper to convert C++ Responses to Python Responses dataclass.
nb::object ToPyResponses(const Responses& responses) {
  nb::object py_responses_class =
      nb::module_::import_(
          "litert_lm.interfaces")
          .attr("Responses");
  auto texts = responses.GetTexts().empty() ? std::vector<std::string>()
                                            : responses.GetTexts();
  auto scores = responses.GetScores();
  auto token_lengths = responses.GetTokenLengths().value_or(std::vector<int>());
  auto token_scores =
      responses.GetTokenScores().value_or(std::vector<std::vector<float>>());
  auto token_ids =
      responses.GetTokenIds().value_or(std::vector<std::vector<int>>());
  auto greedy_token_ids =
      responses.GetGreedyTokenIds().value_or(std::vector<std::vector<int>>());
  auto finish_reasons =
      responses.GetFinishReasons().value_or(std::vector<std::string>());
  return py_responses_class(texts, scores, token_lengths, token_scores,
                            token_ids, greedy_token_ids, finish_reasons);
}

// Note: Consider move to C++ API.
enum class LogSeverity {
  kVerbose = 0,
  kDebug = 1,
  kInfo = 2,
  kWarning = 3,
  kError = 4,
  kFatal = 5,
  kSilent = 1000,
};

// MessageIterator bridges the asynchronous, callback-based C++ API
// (Conversation::SendMessageAsync) to Python's synchronous iterator protocol
// (__iter__ / __next__).
//
// It provides a thread-safe queue where the background C++ inference thread
// pushes generated message chunks. The Python main thread can then safely
// pull these chunks one by one by iterating over this object.
//
// This design keeps the C++ background thread completely free from Python's
// Global Interpreter Lock (GIL), maximizing concurrency and preventing
// deadlocks.
class MessageIterator {
 public:
  MessageIterator() = default;

  MessageIterator(const MessageIterator&) = delete;
  MessageIterator& operator=(const MessageIterator&) = delete;

  void Push(absl::StatusOr<Message> message) {
    absl::MutexLock lock(&mutex_);
    queue_.push_back(std::move(message));
  }

  nlohmann::json Next() {
    absl::StatusOr<Message> message;
    {
      nb::gil_scoped_release release;
      absl::MutexLock lock(&mutex_);
      mutex_.Await(absl::Condition(this, &MessageIterator::HasData));
      message = std::move(queue_.front());
      queue_.pop_front();
    }

    if (!message.ok()) {
      if (absl::IsCancelled(message.status())) {
        throw nb::stop_iteration();
      }
      throw std::runtime_error(message.status().ToString());
    }

    if (!std::holds_alternative<JsonMessage>(*message)) {
      throw std::runtime_error(
          "SendMessageAsync did not return a JsonMessage.");
    }

    auto& json_msg = std::get<JsonMessage>(*message);
    if (json_msg.empty()) {
      throw nb::stop_iteration();
    }

    return static_cast<nlohmann::json>(json_msg);
  }

  bool HasData() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return !queue_.empty();
  }

 private:
  absl::Mutex mutex_;
  std::deque<absl::StatusOr<Message>> queue_ ABSL_GUARDED_BY(mutex_);
};

struct PyBenchmarkInfo {
  double init_time_in_second;
  double time_to_first_token_in_second;
  int last_prefill_token_count;
  double last_prefill_tokens_per_second;
  int last_decode_token_count;
  double last_decode_tokens_per_second;
};

std::optional<int> GetBosTokenIdFromEngineSettings(
    const EngineSettings& engine_settings) {
  const auto& metadata = engine_settings.GetLlmMetadata();
  if (!metadata.has_value() || !metadata->has_start_token() ||
      !metadata->start_token().has_token_ids() ||
      metadata->start_token().token_ids().ids_size() == 0) {
    return std::nullopt;
  }
  return metadata->start_token().token_ids().ids(0);
}

std::vector<std::vector<int>> GetEosTokenIdsFromEngineSettings(
    const EngineSettings& engine_settings) {
  std::vector<std::vector<int>> stop_token_ids;
  const auto& metadata = engine_settings.GetLlmMetadata();
  if (!metadata.has_value()) {
    return stop_token_ids;
  }
  stop_token_ids.reserve(metadata->stop_tokens_size());
  for (const auto& stop_token : metadata->stop_tokens()) {
    if (!stop_token.has_token_ids()) {
      continue;
    }
    stop_token_ids.emplace_back(stop_token.token_ids().ids().begin(),
                                stop_token.token_ids().ids().end());
  }
  return stop_token_ids;
}

absl::StatusOr<std::unique_ptr<Tokenizer>> CreateTokenizerFromEngine(
    const Engine& engine) {
  const auto& model_assets =
      engine.GetEngineSettings().GetMainExecutorSettings().GetModelAssets();
  ASSIGN_OR_RETURN(auto model_resources,
                   BuildLiteRtCompiledModelResources(model_assets));
  return model_resources->GetTokenizer();
}

class PyTokenizer {
 public:
  PyTokenizer(std::unique_ptr<Tokenizer> tokenizer,
              std::optional<int> bos_token_id,
              std::vector<std::vector<int>> eos_token_ids)
      : tokenizer_(std::move(tokenizer)),
        bos_token_id_(bos_token_id),
        eos_token_ids_(std::move(eos_token_ids)) {}

  PyTokenizer(const PyTokenizer&) = delete;
  PyTokenizer& operator=(const PyTokenizer&) = delete;
  PyTokenizer(PyTokenizer&&) = default;
  PyTokenizer& operator=(PyTokenizer&&) = default;

  std::vector<int> Encode(std::string_view text) {
    return VALUE_OR_THROW(tokenizer_->TextToTokenIds(text));
  }

  std::string Decode(const std::vector<int>& token_ids) {
    return VALUE_OR_THROW(tokenizer_->TokenIdsToText(token_ids));
  }

  std::optional<int> GetBosTokenId() const { return bos_token_id_; }

  const std::vector<std::vector<int>>& GetEosTokenIds() const {
    return eos_token_ids_;
  }

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::optional<int> bos_token_id_;
  std::vector<std::vector<int>> eos_token_ids_;
};

class Benchmark {
 public:
  Benchmark(std::string model_path, Backend backend, int prefill_tokens,
            int decode_tokens, std::string cache_dir)
      : model_path_(std::move(model_path)),
        backend_(backend),
        prefill_tokens_(prefill_tokens),
        decode_tokens_(decode_tokens),
        cache_dir_(std::move(cache_dir)) {}

  PyBenchmarkInfo Run() {
    auto model_assets = VALUE_OR_THROW(ModelAssets::Create(model_path_));
    auto settings =
        VALUE_OR_THROW(EngineSettings::CreateDefault(model_assets, backend_));

    if (!cache_dir_.empty()) {
      settings.GetMutableMainExecutorSettings().SetCacheDir(cache_dir_);
    }

    auto& benchmark_params = settings.GetMutableBenchmarkParams();
    benchmark_params.set_num_prefill_tokens(prefill_tokens_);
    benchmark_params.set_num_decode_tokens(decode_tokens_);

    auto engine =
        VALUE_OR_THROW(EngineFactory::CreateDefault(std::move(settings)));

    auto conversation_config =
        VALUE_OR_THROW(ConversationConfig::CreateDefault(*engine));
    auto conversation =
        VALUE_OR_THROW(Conversation::Create(*engine, conversation_config));

    // Trigger benchmark
    nlohmann::json dummy_message = {
        {"role", "user"},
        {"content", "Engine ignore this message in this mode."}};
    (void)VALUE_OR_THROW(conversation->SendMessage(dummy_message));

    auto benchmark_info_cpp = VALUE_OR_THROW(conversation->GetBenchmarkInfo());

    PyBenchmarkInfo result;

    double total_init_time_ms = 0.0;
    for (const auto& phase : benchmark_info_cpp.GetInitPhases()) {
      total_init_time_ms += absl::ToDoubleMilliseconds(phase.second);
    }
    result.init_time_in_second = total_init_time_ms / 1000.0;
    result.time_to_first_token_in_second =
        benchmark_info_cpp.GetTimeToFirstToken();

    int last_prefill_token_count = 0;
    double last_prefill_tokens_per_second = 0.0;
    if (benchmark_info_cpp.GetTotalPrefillTurns() > 0) {
      int last_index =
          static_cast<int>(benchmark_info_cpp.GetTotalPrefillTurns()) - 1;
      auto turn = benchmark_info_cpp.GetPrefillTurn(last_index);
      if (turn.ok()) {
        last_prefill_token_count = static_cast<int>(turn->num_tokens);
      }
      last_prefill_tokens_per_second =
          benchmark_info_cpp.GetPrefillTokensPerSec(last_index);
    }
    result.last_prefill_token_count = last_prefill_token_count;
    result.last_prefill_tokens_per_second = last_prefill_tokens_per_second;

    int last_decode_token_count = 0;
    double last_decode_tokens_per_second = 0.0;
    if (benchmark_info_cpp.GetTotalDecodeTurns() > 0) {
      int last_index =
          static_cast<int>(benchmark_info_cpp.GetTotalDecodeTurns()) - 1;
      auto turn = benchmark_info_cpp.GetDecodeTurn(last_index);
      if (turn.ok()) {
        last_decode_token_count = static_cast<int>(turn->num_tokens);
      }
      last_decode_tokens_per_second =
          benchmark_info_cpp.GetDecodeTokensPerSec(last_index);
    }
    result.last_decode_token_count = last_decode_token_count;
    result.last_decode_tokens_per_second = last_decode_tokens_per_second;

    return result;
  }

 private:
  // Path to the model file.
  std::string model_path_;
  // Hardware backend used for inference.
  Backend backend_;
  // Number of tokens for the prefill phase.
  int prefill_tokens_;
  // Number of tokens for the decode phase.
  int decode_tokens_;
  // Directory for caching compiled model artifacts.
  std::string cache_dir_;
};

NB_MODULE(litert_lm_ext, module) {
  nb::enum_<LogSeverity>(module, "LogSeverity")
      .value("VERBOSE", LogSeverity::kVerbose)
      .value("DEBUG", LogSeverity::kDebug)
      .value("INFO", LogSeverity::kInfo)
      .value("WARNING", LogSeverity::kWarning)
      .value("ERROR", LogSeverity::kError)
      .value("FATAL", LogSeverity::kFatal)
      .value("SILENT", LogSeverity::kSilent)
      .export_values();

  module.def(
      "Engine",
      [](std::string_view model_path, const nb::handle& backend,
         int max_num_tokens, std::string_view cache_dir,
         const nb::handle& vision_backend, const nb::handle& audio_backend,
         std::string_view input_prompt_as_hint) {
        Backend main_backend = ParseBackend(backend);
        std::optional<Backend> vision_backend_opt = std::nullopt;
        if (!vision_backend.is_none()) {
          vision_backend_opt = ParseBackend(vision_backend);
        }
        std::optional<Backend> audio_backend_opt = std::nullopt;
        if (!audio_backend.is_none()) {
          audio_backend_opt = ParseBackend(audio_backend);
        }

        auto model_assets = VALUE_OR_THROW(ModelAssets::Create(model_path));
        auto settings = VALUE_OR_THROW(EngineSettings::CreateDefault(
            model_assets, main_backend, vision_backend_opt, audio_backend_opt));

        settings.GetMutableMainExecutorSettings().SetMaxNumTokens(
            max_num_tokens);
        if (!cache_dir.empty()) {
          settings.GetMutableMainExecutorSettings().SetCacheDir(
              std::string(cache_dir));
        }

        auto engine = VALUE_OR_THROW(
            EngineFactory::CreateDefault(settings, input_prompt_as_hint));

        nb::object py_engine = nb::cast(std::move(engine));
        py_engine.attr("model_path") = model_path;
        SetBackendAttr(py_engine, backend);
        py_engine.attr("max_num_tokens") = max_num_tokens;
        py_engine.attr("cache_dir") = cache_dir;
        return py_engine;
      },
      nb::arg("model_path"), nb::arg("backend") = nb::none(),
      nb::arg("max_num_tokens") = 4096, nb::arg("cache_dir") = "",
      nb::arg("vision_backend") = nb::none(),
      nb::arg("audio_backend") = nb::none(),
      nb::arg("input_prompt_as_hint") = "");

  module.def(
      "set_min_log_severity",
      [](LogSeverity log_severity) {
        struct SeverityMapping {
          absl::LogSeverityAtLeast absl_severity;
          LiteRtLogSeverity litert_severity;
          tflite::LogSeverity tflite_severity;
        };

        static const std::map<LogSeverity, SeverityMapping> mapping = {
            {LogSeverity::kVerbose,
             {absl::LogSeverityAtLeast::kInfo, kLiteRtLogSeverityVerbose,
              tflite::TFLITE_LOG_VERBOSE}},
            {LogSeverity::kDebug,
             {absl::LogSeverityAtLeast::kInfo, kLiteRtLogSeverityDebug,
              tflite::TFLITE_LOG_VERBOSE}},
            {LogSeverity::kInfo,
             {absl::LogSeverityAtLeast::kInfo, kLiteRtLogSeverityInfo,
              tflite::TFLITE_LOG_INFO}},
            {LogSeverity::kWarning,
             {absl::LogSeverityAtLeast::kWarning, kLiteRtLogSeverityWarning,
              tflite::TFLITE_LOG_WARNING}},
            {LogSeverity::kError,
             {absl::LogSeverityAtLeast::kError, kLiteRtLogSeverityError,
              tflite::TFLITE_LOG_ERROR}},
            {LogSeverity::kFatal,
             {absl::LogSeverityAtLeast::kFatal, kLiteRtLogSeverityError,
              tflite::TFLITE_LOG_ERROR}},
            {LogSeverity::kSilent,
             {absl::LogSeverityAtLeast::kInfinity, kLiteRtLogSeveritySilent,
              tflite::TFLITE_LOG_SILENT}},
        };

        auto mapping_it = mapping.find(log_severity);
        const SeverityMapping& severity_mapping =
            (mapping_it != mapping.end()) ? mapping_it->second
                                          : mapping.at(LogSeverity::kSilent);

        absl::SetMinLogLevel(severity_mapping.absl_severity);
        absl::SetStderrThreshold(severity_mapping.absl_severity);
        LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(),
                                   severity_mapping.litert_severity);
        tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
            severity_mapping.tflite_severity);
      },
      nb::arg("log_severity"));

  nb::class_<PyTokenizer>(module, "Tokenizer", nb::dynamic_attr())
      .def("encode", &PyTokenizer::Encode, nb::arg("text"))
      .def("decode", &PyTokenizer::Decode, nb::arg("token_ids"))
      .def_prop_ro("bos_token_id", &PyTokenizer::GetBosTokenId)
      .def_prop_ro("eos_token_ids", &PyTokenizer::GetEosTokenIds);

  nb::class_<Engine>(module, "_Engine", nb::dynamic_attr())
      // Support for Python context managers (with statement).
      // __enter__ returns the object itself.
      .def("__enter__", [](nb::handle self) { return self; })
      // __exit__ immediately destroys the underlying C++ instance to free
      // resources deterministically, instead of waiting for garbage collection.
      .def(
          "__exit__",
          [](nb::handle self, nb::handle exc_type, nb::handle exc_value,
             nb::handle traceback) { nb::inst_destruct(self); },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def(
          "create_conversation",
          [](const nb::object& self, const nb::handle& messages,
             const nb::handle& tools, const nb::handle& tool_event_handler) {
            Engine& engine = nb::cast<Engine&>(self);

            auto builder = ConversationConfig::Builder();
            nb::dict py_tool_map;

            bool has_preface = false;
            JsonPreface json_preface;

            if (!messages.is_none()) {
              json_preface.messages = nb::cast<nlohmann::json>(messages);
              has_preface = true;
            }

            if (!tools.is_none()) {
              nb::object tool_from_function =
                  nb::module_::import_(
                      "litert_lm."
                      "tools")
                      .attr("tool_from_function");

              nlohmann::json json_tools = nlohmann::json::array();
              for (auto tool : nb::cast<nb::list>(tools)) {
                auto tool_obj = tool_from_function(tool);
                auto description = tool_obj.attr("get_tool_description")();
                std::string name =
                    nb::cast<std::string>(description["function"]["name"]);
                py_tool_map[name.c_str()] = tool_obj;
                json_tools.push_back(nb::cast<nlohmann::json>(description));
              }

              json_preface.tools = std::move(json_tools);
              has_preface = true;
            }

            if (has_preface) {
              builder.SetPreface(json_preface);
            }

            auto config = VALUE_OR_THROW(builder.Build(engine));

            auto conversation =
                VALUE_OR_THROW(Conversation::Create(engine, config));

            nb::object py_conversation = nb::cast(std::move(conversation));
            py_conversation.attr("_tool_map") = py_tool_map;
            py_conversation.attr("tool_event_handler") = tool_event_handler;
            if (messages.is_none()) {
              py_conversation.attr("messages") = nb::list();
            } else {
              py_conversation.attr("messages") = messages;
            }
            if (tools.is_none()) {
              py_conversation.attr("tools") = nb::list();
            } else {
              py_conversation.attr("tools") = tools;
            }
            return py_conversation;
          },
          nb::kw_only(), nb::arg("messages") = nb::none(),
          nb::arg("tools") = nb::none(),
          nb::arg("tool_event_handler") = nb::none())
      .def(
          "create_session",
          [](Engine& self, const nb::handle& options) {
            return VALUE_OR_THROW(self.CreateSession(ParseSessionOptions(options)));
          },
          nb::arg("options") = nb::none(),
          "Creates a new session for this engine.")
      .def(
          "get_tokenizer",
          [](const Engine& self) {
            return std::make_unique<PyTokenizer>(
                VALUE_OR_THROW(CreateTokenizerFromEngine(self)),
                GetBosTokenIdFromEngineSettings(self.GetEngineSettings()),
                GetEosTokenIdsFromEngineSettings(self.GetEngineSettings()));
          },
          "Returns the tokenizer associated with this engine.");

  nb::class_<Engine::Session>(module, "Session", nb::dynamic_attr(),
                              "Session is responsible for hosting the "
                              "internal state (e.g. conversation history) of "
                              "each separate interaction with LLM.")
      // Support for Python context managers (with statement).
      // __enter__ returns the object itself.
      .def("__enter__", [](nb::handle self) { return self; })
      // __exit__ immediately destroys the underlying C++ instance to free
      // resources deterministically, instead of waiting for garbage collection.
      .def(
          "__exit__",
          [](nb::handle self, nb::handle exc_type, nb::handle exc_value,
             nb::handle traceback) { nb::inst_destruct(self); },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def(
          "run_prefill",
          [](Engine::Session& self, const std::vector<std::string>& contents) {
            std::vector<InputData> input_data;
            for (const auto& text : contents) {
              input_data.emplace_back(InputText(text));
            }
            STATUS_OR_THROW(self.RunPrefill(input_data));
          },
          nb::arg("contents"),
          "Adds the input prompt/query to the model for starting the "
          "prefilling process. Note that the user can break down their "
          "prompt/query into multiple chunks and call this function multiple "
          "times.")
      .def(
          "run_prefill_token_ids",
          [](Engine::Session& self, const std::vector<int>& token_ids) {
            auto token_id_buffer =
                VALUE_OR_THROW(Tokenizer::TokenIdsToTensorBuffer(token_ids));
            std::vector<InputData> input_data;
            input_data.emplace_back(InputText(std::move(token_id_buffer)));
            STATUS_OR_THROW(self.RunPrefill(input_data));
          },
          nb::arg("token_ids"),
          "Adds already-tokenized input ids to the session prefill state.")
      .def(
          "run_decode",
          [](Engine::Session& self, const nb::handle& options) {
            return ToPyResponses(
                VALUE_OR_THROW(self.RunDecode(ParseDecodeOptions(options))));
          },
          nb::arg("options") = nb::none(),
          "Starts the decoding process for the model to predict the response "
          "based on the input prompt/query added after using run_prefill "
          "function.")
      .def(
          "run_text_scoring",
          [](Engine::Session& self, const std::vector<std::string>& target_text,
             bool store_token_lengths) {
            std::vector<absl::string_view> target_text_views;
            for (const auto& text : target_text) {
              target_text_views.push_back(text);
            }
            return ToPyResponses(VALUE_OR_THROW(
                self.RunTextScoring(target_text_views, store_token_lengths)));
          },
          nb::arg("target_text"), nb::arg("store_token_lengths") = false,
          "Scores the target text after the prefill process is done.")
      .def(
          "run_token_scoring",
          [](Engine::Session& self,
             const std::vector<std::vector<int>>& target_token_ids,
             bool store_token_lengths) {
            return ToPyResponses(VALUE_OR_THROW(
                self.RunTokenIdScoring(target_token_ids, store_token_lengths)));
          },
          nb::arg("target_token_ids"),
          nb::arg("store_token_lengths") = false,
          "Scores the target token ids after the prefill process is done.");

  nb::class_<Conversation>(module, "Conversation", nb::dynamic_attr())
      // Support for Python context managers (with statement).
      // __enter__ returns the object itself.
      .def("__enter__", [](nb::handle self) { return self; })
      // __exit__ immediately destroys the underlying C++ instance to free
      // resources deterministically, instead of waiting for garbage collection.
      .def(
          "__exit__",
          [](nb::handle self, nb::handle exc_type, nb::handle exc_value,
             nb::handle traceback) { nb::inst_destruct(self); },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def("cancel_process", &Conversation::CancelProcess)
      .def(
          "send_message",
          [](nb::object self, const nb::handle& message) {
            Conversation& conversation = nb::cast<Conversation&>(self);
            nlohmann::json current_message = ParseJsonMessage(message);

            nb::dict tool_map;
            if (nb::hasattr(self, "_tool_map")) {
              tool_map = nb::cast<nb::dict>(self.attr("_tool_map"));
            }

            nb::object tool_event_handler = nb::none();
            if (nb::hasattr(self, "tool_event_handler")) {
              tool_event_handler = self.attr("tool_event_handler");
            }

            for (int i = 0; i < kRecurringToolCallLimit; ++i) {
              absl::StatusOr<Message> result =
                  conversation.SendMessage(current_message);
              Message message_variant = VALUE_OR_THROW(std::move(result));

              if (!std::holds_alternative<JsonMessage>(message_variant)) {
                throw std::runtime_error(
                    "SendMessage did not return a JsonMessage.");
              }

              nlohmann::json response = std::get<JsonMessage>(message_variant);

              if (response.contains("tool_calls") &&
                  !response["tool_calls"].empty()) {
                current_message =
                    HandleToolCalls(response, tool_map, tool_event_handler);
              } else {
                return response;
              }
            }
            throw std::runtime_error("Exceeded recurring tool call limit of " +
                                     std::to_string(kRecurringToolCallLimit));
          },
          nb::arg("message"))
      .def(
          "send_message_async",
          [](nb::object self, const nb::handle& message) {
            Conversation& conversation = nb::cast<Conversation&>(self);
            nlohmann::json json_message = ParseJsonMessage(message);
            auto iterator = std::make_shared<MessageIterator>();

            nb::dict tool_map;
            if (nb::hasattr(self, "_tool_map")) {
              tool_map = nb::cast<nb::dict>(self.attr("_tool_map"));
            }

            nb::object tool_event_handler = nb::none();
            if (nb::hasattr(self, "tool_event_handler")) {
              tool_event_handler = self.attr("tool_event_handler");
            }

            if (tool_map.empty() && tool_event_handler.is_none()) {
              struct SimpleCallback {
                std::shared_ptr<MessageIterator> iterator;

                void operator()(absl::StatusOr<Message> message) const {
                  iterator->Push(std::move(message));
                }
              };

              absl::Status status = conversation.SendMessageAsync(
                  json_message, SimpleCallback{iterator});
              if (!status.ok()) {
                std::stringstream error_msg_stream;
                error_msg_stream << "SendMessageAsync failed: " << status;
                throw std::runtime_error(error_msg_stream.str());
              }
              return iterator;
            }

            struct AsyncState {
              int tool_call_count = 0;
              nlohmann::json pending_tool_response = nullptr;
            };
            auto state = std::make_shared<AsyncState>();

            struct Callback {
              nb::object self;
              std::shared_ptr<MessageIterator> iterator;
              nb::dict tool_map;
              nb::object tool_event_handler;
              std::shared_ptr<AsyncState> state;

              void operator()(absl::StatusOr<Message> message) const {
                if (!message.ok()) {
                  iterator->Push(std::move(message));
                  return;
                }

                if (!std::holds_alternative<JsonMessage>(*message)) {
                  iterator->Push(absl::InternalError(
                      "SendMessageAsync did not return a JsonMessage."));
                  return;
                }

                auto& json_msg = std::get<JsonMessage>(*message);

                if (json_msg.contains("tool_calls") &&
                    !json_msg["tool_calls"].empty()) {
                  nb::gil_scoped_acquire acquire;
                  state->pending_tool_response =
                      HandleToolCalls(json_msg, tool_map, tool_event_handler);
                }

                if (json_msg.contains("content")) {
                  iterator->Push(std::move(message));
                } else if (json_msg.empty()) {
                  if (state->pending_tool_response != nullptr) {
                    if (state->tool_call_count >= kRecurringToolCallLimit) {
                      iterator->Push(absl::InternalError(
                          "Exceeded recurring tool call limit of " +
                          std::to_string(kRecurringToolCallLimit)));
                      return;
                    }
                    state->tool_call_count++;
                    nlohmann::json next_message =
                        std::move(state->pending_tool_response);
                    state->pending_tool_response = nullptr;

                    nb::gil_scoped_acquire acquire;
                    Conversation& conv = nb::cast<Conversation&>(self);
                    absl::Status status =
                        conv.SendMessageAsync(next_message, *this);
                    if (!status.ok()) {
                      iterator->Push(status);
                    }
                  } else {
                    iterator->Push(std::move(message));
                  }
                }
              }
            };

            absl::Status status = conversation.SendMessageAsync(
                json_message,
                Callback{self, iterator, tool_map, tool_event_handler, state});

            if (!status.ok()) {
              std::stringstream error_msg_stream;
              error_msg_stream << "SendMessageAsync failed: " << status;
              throw std::runtime_error(error_msg_stream.str());
            }
            return iterator;
          },
          nb::arg("message"));
  // Expose the MessageIterator to Python so that it can be used in a
  // standard `for chunk in stream:` loop. We bind Python's iterator protocol
  // (__iter__ and __next__) to our C++ implementation.
  nb::class_<MessageIterator>(module, "MessageIterator")
      .def("__iter__", [](nb::handle self) { return self; })
      .def("__next__", &MessageIterator::Next);

  module.def(
      "Benchmark",
      [](std::string_view model_path, const nb::handle& backend,
         int prefill_tokens, int decode_tokens, std::string_view cache_dir) {
        auto benchmark = std::make_unique<Benchmark>(
            std::string(model_path), ParseBackend(backend), prefill_tokens,
            decode_tokens, std::string(cache_dir));

        nb::object py_benchmark = nb::cast(std::move(benchmark));
        py_benchmark.attr("model_path") = model_path;
        SetBackendAttr(py_benchmark, backend);
        py_benchmark.attr("prefill_tokens") = prefill_tokens;
        py_benchmark.attr("decode_tokens") = decode_tokens;
        py_benchmark.attr("cache_dir") = cache_dir;
        return py_benchmark;
      },
      nb::arg("model_path"), nb::arg("backend") = nb::none(),
      nb::arg("prefill_tokens") = 256, nb::arg("decode_tokens") = 256,
      nb::arg("cache_dir") = "");

  nb::class_<PyBenchmarkInfo>(module, "BenchmarkInfo",
                              "Data class to hold benchmark information.")
      .def_rw("init_time_in_second", &PyBenchmarkInfo::init_time_in_second,
              "The time in seconds to initialize the engine and the "
              "conversation.")
      .def_rw("time_to_first_token_in_second",
              &PyBenchmarkInfo::time_to_first_token_in_second,
              "The time in seconds to the first token.")
      .def_rw(
          "last_prefill_token_count",
          &PyBenchmarkInfo::last_prefill_token_count,
          "The number of tokens in the last prefill. Returns 0 if there was "
          "no prefill.")
      .def_rw("last_prefill_tokens_per_second",
              &PyBenchmarkInfo::last_prefill_tokens_per_second,
              "The number of tokens processed per second in the last prefill.")
      .def_rw("last_decode_token_count",
              &PyBenchmarkInfo::last_decode_token_count,
              "The number of tokens in the last decode. Returns 0 if there was "
              "no decode.")
      .def_rw("last_decode_tokens_per_second",
              &PyBenchmarkInfo::last_decode_tokens_per_second,
              "The number of tokens processed per second in the last decode.");

  nb::class_<Benchmark>(module, "_Benchmark", nb::dynamic_attr())
      .def("run", &Benchmark::Run);
}

}  // namespace litert::lm
