# Held-out evaluation

The frozen cases were generated with seed `20261030` before any 9B evaluation.
There are 24 arithmetic expressions (addition, subtraction, multiplication,
gcd, exact fractions, floor division, modulo) and 12 8-bit programs. The three
earlier 9B smoke expressions are excluded. The 8-bit program suite includes
the initial state and every reference state through HALT. Both suites have
since been run; see [BENCHMARK.md](../BENCHMARK.md) and
[DYNAMIC_ROM.md](../DYNAMIC_ROM.md).

Frozen file SHA-256: `a4277e1e79229b258319454b237f2f84749c67da3ad1d60f0dcbd3fc6f4e0896`
for arithmetic, `9b47c391cb96c55d176b688cb2bc009bb8e6a607e3498d8e81f260b2e7833f06`
for programs. Regenerate with `python -m eval.cases --out eval/frozen`, but do
not change the frozen files after model evaluation begins.

## Arithmetic comparison

Run on a CUDA host with the patched, pinned `qwenputer` checkout and both the
base and merged Qwen3.5-9B weights. Install the checkout's `hf` extra first.
The same frozen cases, 4,096-token text budget, and greedy decoding are used
for all conditions. The
merged and base-host arms receive the same tool schema and system prompt.
The no-tool base arm gets a by-hand prompt. The base-host arm only executes
the reference calc firmware when the model actually calls `calc`; it does not
inject an answer in advance.

From this lab's directory, replace paths below with paths on the CUDA host:

```sh
python -m eval.run_arithmetic --arm base \
  --model BASE --qwenputer-src QWENPUTER/experimental/riscv \
  --firmware QWENPUTER/experimental/riscv/tests/data/rv/calc-bridge.elf \
  --cases eval/frozen/arithmetic.jsonl --output eval/results/base.jsonl
python -m eval.run_arithmetic --arm base_host \
  --model BASE --qwenputer-src QWENPUTER/experimental/riscv \
  --firmware QWENPUTER/experimental/riscv/tests/data/rv/calc-bridge.elf \
  --cases eval/frozen/arithmetic.jsonl --output eval/results/base_host.jsonl
python -m eval.run_arithmetic --arm merged \
  --model MERGED --qwenputer-src QWENPUTER/experimental/riscv \
  --firmware QWENPUTER/experimental/riscv/tests/data/rv/calc-bridge.elf \
  --cases eval/frozen/arithmetic.jsonl --output eval/results/merged.jsonl --replay
python -m eval.score --arithmetic-cases eval/frozen/arithmetic.jsonl \
  --arithmetic-results eval/results/base.jsonl eval/results/base_host.jsonl \
  eval/results/merged.jsonl
```

Each JSONL record includes exact rational scoring, tool-call count, token
counts, latency, parser source, runtime error, and generated text. The merged
arm also records head disagreement, reference lock error, and optional full
IR/reference replay. A replay error is reported separately and does not erase
the generated answer. Resume checks the arm, model path, token limits, frozen
fixture, and calculator firmware before skipping existing case IDs. Tool calls
count emitted `</tool_call>` tokens; completed calls count reference firmware
runs. Scoring checks that both tool arms received identical prompt token IDs.
The summary
reports missing cases explicitly; rates use
only completed case records. It reports both any exact parsed answer and the
stricter exact answer in `\boxed{}`. Token-limit hits and parse failures are
counted separately. A score on this synthetic 24-case set is a bounded
comparison, not a general benchmark result.
The 4,096-token limit counts forced tool-response text in the merged arm but
only model-generated text in the host arm. Both arms hit the limit on the same
one case, yet the numerical limits are not strictly equivalent after a call.

## 8-bit program evaluation

The model-side 8-bit adapter has a newline-delimited JSON contract. It reads
`{"id": ..., "program_hex": ..., "max_steps": ...}` and replies with
`{"id": ..., "trace": [...]}`. Each trace includes the initial state and one
state after every instruction, ending with HALT. Each state has `pc`, `acc`,
`ram_hex` (16 bytes), and `halted`. The adapter must obtain its trace from the
actual model path; it must not import or call `gdn_cpu.reference`.

```sh
python -m eval.run_programs --cases eval/frozen/programs.jsonl \
  --output eval/results/programs.jsonl --adapter-cmd 'python path/to/model_adapter.py'
python -m eval.score --arithmetic-cases eval/frozen/arithmetic.jsonl \
  --program-cases eval/frozen/programs.jsonl \
  --program-results eval/results/programs.jsonl
```

The adapter contract remains available for other models. This lab also runs
the frozen cases directly through `gdn_cpu.qwen8_dynamic_eval` and
`gdn_cpu.byte9b_check`, both of which save generated token IDs and inspect
actual recurrent cache state. The [CUDA dynamic result](../results/dynamic16_cuda.json)
and [9B merged result](../results/byte9b_dynamic_programs.json) passed all
12 frozen cases.
