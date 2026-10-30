# Byte CPU in stock Qwen3.5 layers

`gdn_cpu/qwen8.py` compiles this project's 8-bit ISA into the generic Qwen3.5
circuit backend from [Qwenputer](https://github.com/divagr18/qwenputer). The
ISA and transition graph are ours; the graph scheduler, exact gate gadgets,
weight emitter, and small model builder are reused from Qwenputer. The model
is a stock `Qwen3_5ForCausalLM` with no trained weights.

Each token encodes the current 8-bit PC, 8-bit accumulator, and halt bit. The
model predicts the next state token under greedy generation, so there is no
host step function during inference. A Gated DeltaNet head holds 16 bytes of
writable RAM and performs addressed reads and overwrites. The program ROM is
fixed in the compiled MLP weights. Every instruction takes one generated
token. The padded token at the start initializes the model's cache.

The graph uses 11 circuit layers, 1 Gated DeltaNet head, 451 FFN units, and
386 residual graph dimensions. The standalone Qwen model has 12 layers
(including an inert full-attention cache layer) and hidden size 448. It has
131,072 state tokens plus a zero padding row.

## Reproduce

From Qwenputer's `experimental/riscv` directory with its package installed:

```sh
PYTHONPATH=/absolute/path/to/gdn-cpu-lab uv run --extra hf --no-default-groups \
  python -m unittest discover -s /absolute/path/to/gdn-cpu-lab/tests \
  -p test_qwen8_ir.py -v
PYTHONPATH=/absolute/path/to/gdn-cpu-lab uv run --extra hf --no-default-groups \
  python -m gdn_cpu.qwen8_check --program 1001200230004000ff --steps 6
```

The IR differential test covers six named programs and 32 seeded straight-line
programs, including branch taken and not taken, arithmetic wraparound, RAM
overwrite and load, and halt. The local stock Transformers model with BF16
weights and reference PyTorch Gated DeltaNet kernels matched the independent
emulator on these observed traces:

| Program | Behavior | Exact generated steps |
| --- | --- | ---: |
| `1001200230004000ff` | LDI, ADD, STA, LDA, HALT | 6/6 |
| `10015008109900002001ff` | Taken JNZ, ADD, HALT | 4/4 |
| `10073002100930024002ff` | Overwrite RAM address 2, load, HALT | 7/7 |

The overwrite program also passed 6/6 generated steps on Apple M4 MPS. The
test ran the stock Transformers reference recurrence on MPS, not the optional
optimized CUDA kernel.

These are deterministic smoke checks, not a held-out benchmark or a claim
that the language model chooses to use the machine. The local Transformers
run used reference PyTorch kernels; optimized GPU kernels still need a separate
check. The current graph has a fixed compiled program and covers valid traces
whose PC stays below 256. Invalid opcode, missing operand, bad RAM address,
and PC overflow remain the reference emulator's explicit errors. PC,
accumulator, and halt status travel in model-generated tokens; RAM is in
recurrent Gated DeltaNet state. This is a different state layout from the
dependency-free circuit proof, which keeps all registers in recurrent state.

## Recurrent register variant

`gdn_cpu/qwen8_recurrent.py` puts PC, accumulator, and halt status in a second
Gated DeltaNet head. The model alternates its own state token and `TICK` token.
The state token writes the register head; `TICK` has no register payload, so
the instruction decoder reads those registers from recurrent state. RAM stays
in its separate Gated DeltaNet head. No host instruction loop supplies either
token. The model still expresses the next state in its generated trace token,
which is how autoregressive output carries the computed state into the next
time step. The ROM remains fixed at graph build time.

```sh
PYTHONPATH=/absolute/path/to/gdn-cpu-lab uv run --extra hf --no-default-groups \
  python -m gdn_cpu.qwen8_check --recurrent \
  --program 1001200230004000ff --steps 5
```

That stock Qwen model generated 10/10 alternating tokens exactly on the
local CPU. The overwrite program generated 12/12 alternating tokens on both
CPU and Apple M4 MPS. For that program, `qwen8_check` also inspected the
actual Transformers recurrent cache after every `TICK` and found all 16 RAM
bytes equal to the independent emulator at each step. The IR test checks
both heads with its kernel-equivalent read calculation and compares all RAM
bytes after every instruction across six named programs. This variant has
11 circuit layers, 2 Gated DeltaNet heads, 456 FFN units, and 407 graph
dimensions; its standalone model remains hidden size 448 with 12 total
layers. A 12-token MPS run also verified that a halted state remains inert.
The local runs used the stock Transformers reference recurrence.

## Frozen 12-program evaluation

`eval/byte12.json` contains 12 programs, 148 expected generated tokens, and
final RAM snapshots frozen from the independent emulator. The fixture was
generated before running the suite. `gdn_cpu/qwen8_eval.py` does not import
the emulator. It builds a new recurrent Qwen model per program, uses stock
greedy `generate()`, and compares the generated tokens and actual recurrent
cache RAM with the frozen records.

```sh
PYTHONPATH=/absolute/path/to/gdn-cpu-lab uv run --extra hf --no-default-groups \
  python -m gdn_cpu.qwen8_eval \
  --fixture /absolute/path/to/gdn-cpu-lab/eval/byte12.json \
  --out /absolute/path/to/gdn-cpu-lab/results/byte12_mps.json \
  --device mps
```

The [MPS result](results/byte12_mps.json) reports 12/12 programs and
148/148 generated tokens exact, with final RAM exact in all 12. These cases
test the standalone byte CPU. They do not measure language-model tool
selection or compare base 9B with a host tool.

The [CUDA result](results/byte12_cuda.json) used the same fixture on an RTX
PRO 6000 Blackwell with PyTorch 2.14.1, Transformers 5.19.0, and BF16. It
also passed 12/12 programs, all 148 tokens, and all 12 final RAM snapshots.
The optional `flash-linear-attention` and `causal-conv1d` packages were absent,
so this CUDA run used Transformers' reference kernels.
