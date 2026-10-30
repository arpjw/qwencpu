# Reusable program ROM in Qwen3.5 layers

`gdn_cpu.qwen8_dynamic` accepts bytecode as input tokens and stores it in
Gated DeltaNet state. The ISA, register head, RAM head, and generated
state/`TICK` protocol are the same as the recurrent byte CPU. The program
bytes are no longer compiled into the MLP weights. One set of model weights
can run different programs on separate, fresh inference caches.

The caller supplies raw token IDs in this order:

1. `PAD` (`131073`).
2. One `LOAD(address, byte)` token per program byte, in address order. Its ID
   is `131074 + 256*address + byte`.
3. The initial state token, normally `0` for PC 0, accumulator 0, not halted.

Greedy generation then alternates `TICK` (`131072`) and a model-generated
state token. Each `TICK` executes one instruction. The CPU's 16-byte RAM and
PC, accumulator, and halt bit are in Gated DeltaNet state. A fresh cache is
used for each program; loading a shorter program into a cache that contains
an earlier program would leave old bytes in untouched ROM rows.

The full 256-byte circuit has four ROM heads, two for opcode lookup and two
for operand lookup, plus one register head and one RAM head. It has 20 circuit
layers, six Gated DeltaNet heads, and 1,773 graph dimensions. The standalone
stock Qwen3.5 model uses hidden size 1,792 and 21 total layers. The
16-byte configuration uses one opcode and one operand ROM head, for four
heads total, hidden size 384, and the same 20 circuit layers. Its 135,170
raw tokens include state, `TICK`, `PAD`, and every address/byte `LOAD` pair.

## Checks completed

The full 256-byte graph passed an exact BF16 IR evaluation for three different
programs using one graph, including a branch to address 127 whose immediate
operand is at 128 and another that executes at addresses 254 and 255. The IR
test runs the backend's kernel-equivalent read calculation. This is a graph
check, not a stock-model run.

The 16-byte **stock Qwen3.5 model** ran the 12 previously frozen programs
with one model object and shared weights on Apple M4 MPS. All 124 generated
tokens matched the reference traces. All 62 per-instruction RAM snapshots
matched, and both actual Transformers ROM cache copies contained the exact
loaded program bytes for every case. The CUDA run also passed all 12 cases,
124 generated tokens, 62 RAM snapshots, and both ROM copies. See
[MPS](results/dynamic16_mps.json) and [CUDA](results/dynamic16_cuda.json) results.
The cases file SHA-256 is
`9b47c391cb96c55d176b688cb2bc009bb8e6a607e3498d8e81f260b2e7833f06`.
The [optimized CUDA run](results/dynamic16_fla_cuda.json) also passed all 12
cases using Flash Linear Attention's Gated DeltaNet kernels. Causal
convolution still used Transformers' reference path.

Two additional high-address cases were frozen before a stock-model run in
[dynamic256.jsonl](eval/frozen/dynamic256.jsonl), SHA-256
`184fed28d5f5130cd1bc4abd071bdc2dd057551756adba097abf2ff77a4b0975`.
One crosses the 127/128 ROM bank boundary; the other branches to 254 and
halts at 255. The full 256-byte **stock Qwen3.5 model** ran both cases with
one model object and shared weights on Apple M4 MPS. All 18 generated tokens
matched the frozen traces; ROM contents in both actual Transformers cache
copies and all nine per-instruction RAM snapshots matched. See
[dynamic256_mps.json](results/dynamic256_mps.json). CUDA also passed both
cases, all 18 generated tokens, all nine RAM snapshots, and both ROM copies;
see [dynamic256_cuda.json](results/dynamic256_cuda.json). These stock-model
runs used Transformers' reference Gated DeltaNet kernels. The
[optimized CUDA run](results/dynamic256_fla_cuda.json) also passed both cases.
The [pretrained 9B merge](BYTE9B.md) passed both frozen suites with reference
and optimized Gated DeltaNet kernels; it preserves a separate learned text
output head and adds a text handoff limitation described there.

## Reproduce

From this directory, with Qwenputer's `experimental/riscv` package on
`PYTHONPATH` and its `hf` dependencies installed:

```sh
python -m unittest tests.test_qwen8_dynamic_ir -v
python -m gdn_cpu.qwen8_dynamic_eval \
  --cases eval/frozen/programs.jsonl --out results/dynamic16_mps.json \
  --device mps --rom-size 16
python -m gdn_cpu.qwen8_dynamic_eval \
  --cases eval/frozen/dynamic256.jsonl --out results/dynamic256_mps.json \
  --device mps --rom-size 256
python -m gdn_cpu.qwen8_dynamic_eval \
  --cases eval/frozen/programs.jsonl --out results/dynamic16_cuda.json \
  --device cuda --rom-size 16
python -m gdn_cpu.qwen8_dynamic_eval \
  --cases eval/frozen/dynamic256.jsonl --out results/dynamic256_cuda.json \
  --device cuda --rom-size 256
```

The model handles valid, halting bytecode within the selected ROM size.
Unloaded ROM rows read as zero; malformed instruction, operand, address,
and out-of-program errors remain the reference emulator's responsibility.
The loader accepts raw IDs, so it is not yet a natural-language tool
interface or a text tokenizer extension. The Qwenputer generic IR, gate
library, and weight emitter are reused as the compiler backend.
