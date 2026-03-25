# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for litert-lm models."""

import dataclasses
import glob
import importlib.util
import inspect
import json
import os
import pathlib
import readline  # pylint: disable=unused-import
import traceback
import litert_lm

try:
  # pylint: disable=g-import-not-at-top
  from litert_lm.adb import adb_benchmark
  from litert_lm.adb import adb_engine

  _HAS_ADB = True
except ImportError:
  _HAS_ADB = False


def load_preset(preset: str):
  """Loads a preset file and returns the tools and messages."""
  print(f"Loading preset from {preset}:")
  if not os.path.exists(preset):
    print(f"Preset file not found: {preset}")
    return None, None

  spec = importlib.util.spec_from_file_location("user_tools", preset)
  if not spec or not spec.loader:
    print(f"Failed to load tools from {preset}")
    return None, None

  user_tools = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(user_tools)

  tools = getattr(user_tools, "tools", None)
  if tools is None:
    tools = [
        obj
        for name, obj in inspect.getmembers(user_tools, inspect.isfunction)
        if obj.__module__ == "user_tools"
    ]

  messages = None
  system_instruction = getattr(user_tools, "system_instruction", None)
  if system_instruction:
    print(f"- System instruction: {system_instruction}")
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}],
    }]

  print("- Tools:")
  for tool in tools:
    print(f"  - {getattr(tool, "__name__", str(tool))}")

  return tools, messages


_GREEN = "\033[92m"
_RESET = "\033[0m"


class LoggingToolEventHandler(litert_lm.ToolEventHandler):
  """Log tool call and tool response events."""

  def approve_tool_call(self, tool_call):
    """Logs a tool call."""
    print(f"{_GREEN}[tool_call] {json.dumps(tool_call['function'])}{_RESET}")
    return True

  def process_tool_response(self, tool_response):
    """Logs a tool response."""
    print(f"{_GREEN}[tool_response] {json.dumps(tool_response)}{_RESET}")
    return tool_response


def _parse_backend(backend: str) -> litert_lm.Backend:
  """Parses the backend string and returns the corresponding Backend enum."""
  backend_lower = backend.lower()
  if backend_lower == "gpu":
    return litert_lm.Backend.GPU
  return litert_lm.Backend.CPU


@dataclasses.dataclass
class Model:
  """Represents a LiteRT-LM model.

  Attributes:
    model_id: The ID of the model.
    model_path: The local path to the model file.
  """

  model_id: str
  model_path: str

  def exists(self) -> bool:
    """Returns True if the model file exists locally."""
    return os.path.isfile(self.model_path)

  def to_str(self) -> str:
    """Returns a string representation of the model."""
    return self.model_id

  def run_interactive(
      self,
      is_android: bool = False,
      backend: str = "cpu",
      preset: str | None = None,
      prompt: str | None = None,
  ):
    """Runs the model interactively or with a single prompt.

    Args:
      is_android: Whether to run the model on an Android device via ADB.
      backend: The backend to use (cpu or gpu).
      preset: Path to a Python file containing tool functions and system
        instructions.
      prompt: A single prompt to run once and exit.
    """
    if not self.exists():
      print(f"Could not find {self.to_str()} locally in {self.model_path}.")
      return

    if not prompt:
      print(f"Loading model {self.to_str()}...")
    try:
      backend_val = _parse_backend(backend)

      tools = None
      messages = None
      if preset:
        tools, messages = load_preset(preset)
        if tools is None:
          return

      handler = LoggingToolEventHandler() if tools else None

      if is_android:
        if not _HAS_ADB:
          raise ImportError("litert_lm.adb dependencies are not available.")
        engine_cm = adb_engine.AdbEngine(self.model_path, backend=backend_val)
      else:
        engine_cm = litert_lm.Engine(self.model_path, backend=backend_val)

      with (
          engine_cm as engine,
          engine.create_conversation(
              tools=tools, messages=messages, tool_event_handler=handler
          ) as conversation,
      ):
        if prompt:
          self._execute_prompt(conversation, prompt)
          return

        print(
            "Model loaded. Type your prompts and press Enter. Type 'exit' to"
            " quit."
        )

        while True:
          try:
            user_prompt = input("> ")
            if user_prompt.lower() == "exit":
              break
            if not user_prompt:
              continue

            self._execute_prompt(conversation, user_prompt)

          except EOFError:
            break
          except KeyboardInterrupt:
            # Catch Ctrl+C at the input prompt
            print()
            continue
          except Exception:  # pylint: disable=broad-exception-caught
            print("Error during inference")
            traceback.print_exc()

        print("Model closed.")

    except Exception:  # pylint: disable=broad-exception-caught
      print("An error occurred")
      traceback.print_exc()

  def _execute_prompt(self, conversation, prompt):
    """Executes a single prompt and prints the result."""
    stream = conversation.send_message_async(prompt)
    try:
      for chunk in stream:
        content_list = chunk.get("content", [])
        for item in content_list:
          if item.get("type") == "text":
            print(item.get("text", ""), end="", flush=True)
      print()
    except KeyboardInterrupt:
      conversation.cancel_process()
      # Empty the iterator queue.
      # This ensures we don't throw away StopIteration.
      for _ in stream:
        pass
      print("\n[Generation cancelled]")

  def benchmark(
      self,
      prefill_tokens: int = 256,
      decode_tokens: int = 256,
      is_android: bool = False,
      backend: str = "cpu",
  ):
    """Benchmarks the model.

    Args:
      prefill_tokens: The number of tokens to prefill.
      decode_tokens: The number of tokens to decode.
      is_android: Whether to run the benchmark on an Android device via ADB.
      backend: The backend to use (cpu or gpu).
    """
    if not self.exists():
      print(f"Could not find {self.to_str()} locally in {self.model_path}.")
      return

    try:
      backend_val = _parse_backend(backend)

      if is_android:
        if not _HAS_ADB:
          raise ImportError("litert_lm.adb dependencies are not available.")
        benchmark_obj = adb_benchmark.AdbBenchmark(
            self.model_path,
            backend=backend_val,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            cache_dir=":nocache",
        )
      else:
        benchmark_obj = litert_lm.Benchmark(
            self.model_path,
            backend=backend_val,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            cache_dir=":nocache",
        )

      print(f"Benchmarking model: {self.to_str()} ({self.model_path})")
      print(f"Number of tokens in prefill: {prefill_tokens}")
      print(f"Number of tokens in decode : {decode_tokens}")
      print(f"Backend                    : {backend}")
      if is_android:
        print("Target                     : Android")

      result = benchmark_obj.run()

      print("----- Results -----")
      print(
          f"Prefill speed:        {result.last_prefill_tokens_per_second:.2f}"
          " tokens/s"
      )
      print(
          f"Decode speed:         {result.last_decode_tokens_per_second:.2f}"
          " tokens/s"
      )
      print(f"Init time:            {result.init_time_in_second:.4f} s")
      print(
          f"Time to first token:  {result.time_to_first_token_in_second:.4f} s"
      )

    except Exception:  # pylint: disable=broad-exception-caught
      print("An error occurred during benchmarking")
      traceback.print_exc()

  @classmethod
  def get_all_models(cls):
    """Returns a list of all locally available models."""
    model_paths = glob.glob(
        "*/model.litertlm",
        root_dir=get_converted_models_base_dir(),
        recursive=True,
    )

    return [
        Model.from_model_id(
            path.removesuffix("/model.litertlm").replace("--", "/")
        )
        for path in model_paths
    ]

  @classmethod
  def from_model_reference(cls, model_reference):
    """Creates a Model instance from a model reference."""
    if os.path.exists(model_reference):
      return cls.from_model_path(model_reference)
    else:
      # assume the reference is model_id
      return cls.from_model_id(model_reference)

  @classmethod
  def from_model_path(cls, model_path):
    """Creates a Model instance from a model path."""
    return cls(
        model_id=os.path.basename(model_path),
        model_path=os.path.abspath(model_path),
    )

  @classmethod
  def from_model_id(cls, model_id):
    """Creates a Model instance from a model ID."""
    return cls(
        model_id=model_id,
        model_path=os.path.join(
            get_converted_models_base_dir(),
            model_id.replace("/", "--"),
            "model.litertlm",
        ),
    )


# Just to use the huggingface convention. Likely to change.
def model_id_dir_name(model_id):
  """Converts a model ID to a directory name."""
  return model_id.replace("/", "--")


# ~/.litert-lm/models
def get_converted_models_base_dir():
  """Gets the base directory for all converted models."""
  return os.path.join(os.path.expanduser("~"), ".litert-lm", "models")


# ~/.litert-lm/models/<model_id>
def get_model_dir(model_id):
  """Gets the model directory for a given model ID."""
  return os.path.join(
      get_converted_models_base_dir(),
      model_id_dir_name(model_id),
  )
