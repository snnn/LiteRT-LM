## Evaluation and Benchmarking

This guide shows how to benchmark a `.litertlm` bundle with LiteRT-LM and how
to evaluate model quality with `lm-evaluation-harness` (`lm-eval`).

The examples below assume:

- this repo is checked out at `$HOME/src/LiteRT-LM`
- `lm-evaluation-harness` is checked out at
  `$HOME/src/lm-evaluation-harness`
- a model bundle is available as `MODEL_PATH`

For example:

```bash
export MODEL_PATH=/data/bt/models/litert-community/Qwen3-0.6B/Qwen3-0.6B.litertlm
```

### Prerequisites

Build the benchmark binary and the Python wheel.

On Linux, build with `--config=linux_libcxx` explicitly so the host binary,
Python extension, and runtime DSOs all use `libc++`. This is required for
Linux local evaluation flows that share C++ objects with LiteRT GPU delegates.

Android and Apple platforms already use `libc++` by default, so this extra
config is not needed there.

Linux:

```bash
bazelisk build -c opt --config=linux_libcxx \
  //runtime/engine:litert_lm_advanced_main
bazelisk build -c opt --config=linux_libcxx \
  //python/litert_lm:litert_lm_wheel
```

Other platforms:

```bash
bazelisk build -c opt //runtime/engine:litert_lm_advanced_main
bazelisk build -c opt //python/litert_lm:litert_lm_wheel
```

Create a `uv` environment for `lm-eval` and install the LiteRT-LM wheel plus
the local harness dependencies:

```bash
uv venv /tmp/litert_lm_eval_env --python 3.13
uv pip install -p /tmp/litert_lm_eval_env/bin/python \
  -e $HOME/src/lm-evaluation-harness
uv pip install -p /tmp/litert_lm_eval_env/bin/python \
  $HOME/src/LiteRT-LM/bazel-bin/python/litert_lm/*.whl
```

If the harness needs extra model dependencies for an HF baseline, install them
with `uv` into the same environment.

On Linux, the installed wheel's main extension dynamically links to
`libLiteRt.so`. By default it resolves to the wheel-packaged copy under the
installed `litert_lm/` directory. If you need to use a different runtime or
delegate build, put the intended directory first in `LD_LIBRARY_PATH`.

### Run a LiteRT-LM Benchmark

Use `litert_lm_advanced_main` for runtime benchmarking.

```bash
bazel-bin/runtime/engine/litert_lm_advanced_main \
  --model_path="${MODEL_PATH}" \
  --backend=cpu \
  --use_session=true \
  --async=false \
  --input_prompt='Benchmark prompt.' \
  --benchmark \
  --benchmark_prefill_tokens=1024 \
  --benchmark_decode_tokens=128 \
  --max_num_tokens=4096 \
  --num_iterations=1
```

Important notes:

- For fixed-shape prefill/decode bundles, pass `--max_num_tokens` explicitly.
  If it is omitted, the benchmark may auto-set it from the benchmark sizes
  instead of the model capacity.
- On Linux GPU runs with local delegate builds, make sure the intended runtime
  DSOs are ahead of the system defaults in `LD_LIBRARY_PATH`.

The benchmark output reports:

- initialization time
- time to first token
- prefill speed
- decode speed

### Run `lm-eval` Against a `.litertlm` Bundle

On Linux, prefer the wheel-based `lm_eval` flow below. This is the path that
has been validated locally with published and locally built `.litertlm`
bundles.

Linux GPU example:

```bash
LD_LIBRARY_PATH=$HOME/src/LiteRT-LM/prebuilt/linux_x86_64:${LD_LIBRARY_PATH} \
PYTHONPATH=$HOME/src/lm-evaluation-harness \
  /tmp/litert_lm_eval_env/bin/python -m lm_eval \
  --model litert-lm \
  --model_args pretrained="${MODEL_PATH}",backend=gpu,prompt_mode=raw \
  --tasks arc_easy \
  --batch_size 1 \
  --limit 100 \
  --output_path /tmp/litert_eval
```

Linux CPU example:

```bash
PYTHONPATH=$HOME/src/lm-evaluation-harness \
  /tmp/litert_lm_eval_env/bin/python -m lm_eval \
  --model litert-lm \
  --model_args pretrained="${MODEL_PATH}",backend=cpu,prompt_mode=raw,num_cpu_threads=16 \
  --tasks piqa,arc_easy,winogrande \
  --batch_size 1 \
  --limit 100 \
  --output_path /tmp/litert_eval_cpu
```

Recommended starting task set for regression checks:

- `piqa`
- `arc_easy`
- `winogrande`
- `hellaswag`
- `lambada_openai`
- `wikitext`

Recommended starting flags:

- `--batch_size 1`
- `--limit 100` for a screening pass
- `--prompt_mode raw` so the harness owns prompt formatting

Notes:

- The optimized loglikelihood path currently requires `prompt_mode=raw`.
- For fixed-shape bundles, pass `max_length=...` in `--model_args` when you
  need an explicit override that matches the bundle's `max_num_tokens`.
- On Linux GPU runs, set `LD_LIBRARY_PATH` explicitly when you need a specific
  runtime or delegate build.
- On Android and Apple platforms, use the same wheel-based `lm_eval` flow, but
  the extra Linux build flag `--config=linux_libcxx` is not needed when
  building the wheel.

### Bazel Eval Wrapper

LiteRT-LM also provides a Bazel wrapper:

- `//python/litert_lm_eval:litert_lm_eval`

This wrapper forwards to `lm-evaluation-harness` and uses the single supported
model name:

- `litert-lm`

On Linux, prefer the wheel-based flow above. The Bazel wrapper may still be
useful on platforms where the runfiles/runtime linkage is already known-good.

### Compare Against a Hugging Face Baseline

Run the same tasks against the original HF model in the same `lm-eval`
environment. The LiteRT-LM command below is the Linux wheel form:

```bash
PYTHONPATH=$HOME/src/lm-evaluation-harness \
  /tmp/litert_lm_eval_env/bin/python -m lm_eval \
  --model litert-lm \
  --model_args pretrained="${MODEL_PATH}",backend=cpu,prompt_mode=raw \
  --tasks piqa,arc_easy,winogrande \
  --batch_size 1 \
  --limit 100 \
  --output_path /tmp/litert_eval_reference

PYTHONPATH=$HOME/src/lm-evaluation-harness \
  /tmp/litert_lm_eval_env/bin/python -m lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-0.6B,trust_remote_code=True,dtype=auto \
  --tasks piqa,arc_easy,winogrande \
  --batch_size 1 \
  --limit 100 \
  --output_path /tmp/hf_eval
```

Compare the saved `results_*.json` files from the HF and LiteRT-LM runs.

### Log Per-Sample Outputs for Debugging

When a task regresses, rerun it with `--log_samples` and a dedicated
`--output_path`. The command below is the Linux wheel form:

```bash
PYTHONPATH=$HOME/src/lm-evaluation-harness \
  /tmp/litert_lm_eval_env/bin/python -m lm_eval \
  --model litert-lm \
  --model_args pretrained="${MODEL_PATH}",backend=cpu,prompt_mode=raw \
  --tasks winogrande \
  --batch_size 1 \
  --limit 100 \
  --output_path /tmp/winogrande_samples \
  --log_samples
```

This writes:

- `results_<timestamp>.json`
- `samples_<task>_<timestamp>.jsonl`

The sample JSONL includes:

- `doc_id`
- the original dataset document
- model arguments per choice
- per-choice scores
- per-sample metrics such as `acc`

This is the easiest way to inspect which examples changed between two bundles.

### Suggested Debug Workflow

1. Run `litert_lm_advanced_main` on CPU to compare raw runtime speed.
2. Run `lm-eval` on a small fixed task set with `--limit 100`.
3. If quality differs, rerun the affected task with `--log_samples`.
4. Compare:
   - old bundle vs new bundle
   - full-precision bundle vs quantized bundle
   - LiteRT bundle vs HF baseline

For multiple-choice tasks such as Winogrande, focus on:

- which `doc_id`s flipped
- the score margin between the correct and incorrect choices
- whether the full-precision rerun agrees with the quantized bundle or with the
  baseline bundle
