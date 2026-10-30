# Rebuild the 9B CPU lab from scratch

These commands rebuild the code path and checkpoints. The lab contains code,
frozen fixtures, a patch, and small results, but no pretrained model weights.
The tested GPU was a 96 GB RTX PRO 6000 Blackwell Server Edition with a 150 GB
temporary volume. The base 9B snapshot occupied about 19 GB, the RV32 merge
about 35 GB, and the dynamic byte-CPU merge about 34 GB. Keep enough RAM and
storage for simultaneous source and output tensors while merging.

## 1. Get pinned source and weights

Clone [Qwenputer](https://github.com/divagr18/qwenputer), then use this lab's
RV32 port patch. The patch is needed for the separate calculator benchmark;
the byte CPU uses the checkout's generic Qwen3.5 compiler backend.

```sh
git clone https://github.com/divagr18/qwenputer.git
cd qwenputer
git checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c
git apply /path/to/gdn-cpu-lab/qwenputer-9b-port.patch
cd experimental/riscv
uv sync --extra hf --no-dev
uv run --extra hf --no-dev hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir /path/to/Qwen3.5-9B
```

The GPU runs used Python 3.12.3, PyTorch 2.14.1+cu130, and Transformers
5.19.0. Check those versions in the environment before comparing results.
For optimized Gated DeltaNet checks, install `flash-linear-attention==0.6.0`,
`fla-core==0.6.0`, and `einops`. Causal convolution still uses the reference
implementation unless `causal_conv1d` is installed separately.

## 2. Check and merge the independent byte CPU

Set `LAB` to this deliverable directory and `SRC` to the checkout's
`experimental/riscv` directory. Use a fresh shell and your actual paths:

```sh
export LAB=/path/to/gdn-cpu-lab
export SRC=/path/to/qwenputer/experimental/riscv
export PYTHONPATH="$SRC:$LAB"
python -m unittest discover -s "$LAB/tests" -v
python -m gdn_cpu.qwen8_dynamic_eval \
  --cases "$LAB/eval/frozen/programs.jsonl" \
  --out "$LAB/results/dynamic16_cuda.json" --device cuda --rom-size 16
python -m gdn_cpu.qwen8_dynamic_eval \
  --cases "$LAB/eval/frozen/dynamic256.jsonl" \
  --out "$LAB/results/dynamic256_cuda.json" --device cuda --rom-size 256
python -m gdn_cpu.byte9b_merge /path/to/Qwen3.5-9B \
  --out /path/to/Qwen3.5-9B-byte-dynamic256 --mode dynamic --rom-size 256
```

The merge fits anti-sink keys by default and preserves the 9B checkpoint's
untied text output head. One merged checkpoint accepts multiple programs via
raw `LOAD` token IDs. These IDs are supplied by the code, not by the copied
text tokenizer. Validate both frozen suites against the independent emulator
and read the actual recurrent ROM and RAM cache:

```sh
python -m gdn_cpu.byte9b_check /path/to/Qwen3.5-9B-byte-dynamic256 \
  --fixture "$LAB/eval/frozen/programs.jsonl" \
  --out "$LAB/results/byte9b_dynamic_programs.json"
python -m gdn_cpu.byte9b_check /path/to/Qwen3.5-9B-byte-dynamic256 \
  --fixture "$LAB/eval/frozen/dynamic256.jsonl" \
  --out "$LAB/results/byte9b_dynamic_highbank.json"
python -m gdn_cpu.byte9b_textcheck /path/to/Qwen3.5-9B \
  /path/to/Qwen3.5-9B-byte-dynamic256 \
  --out "$LAB/results/byte9b_dynamic_textcheck.json"
```

The [byte CPU report](BYTE9B.md) gives the exact observed scope and the
post-CPU text handoff limitation.

## 3. Run text-selected bytecode with the same checkpoint

The host supplies a tool schema, parses the call, and routes the CPU's raw
token IDs through a fresh cache of the same merged model. It returns the
model-generated result in a normal text tool response. The eight cases were
frozen before this 9B bridge run:

```sh
python -m gdn_cpu.byte9b_bridge_eval \
  /path/to/Qwen3.5-9B-byte-dynamic256 \
  --fixture "$LAB/eval/frozen/byte_bridge.jsonl" \
  --out "$LAB/results/byte9b_bridge_9b.jsonl"
```

## 4. Reproduce the separate RV32 calculator comparison

Build the patched published calculator checkpoint, then run all three arms
using the frozen 24 arithmetic cases. The exact commands and scoring contract
are in [eval/README.md](eval/README.md); the observed result and limits are
in [BENCHMARK.md](BENCHMARK.md). The RV32 calculator and the independent byte
CPU are separate circuits and should not be conflated.

Delete paid compute and storage after saving the small JSON and log files.
The source and output model checkpoints are reproducible from the pinned
inputs, but they are not included in this deliverable.
