# Toy text-to-byte-CPU bridge

`gdn_cpu.byte_bridge` connects two models. Base Qwen3.5-9B reads a question
and decides whether to call the `byte_add7(x)` tool. A separate standalone
stock Qwen3.5 model executes the fixed byte program `20 07 ff`, meaning
ADD immediate 7, then HALT. Both use greedy decoding. The CPU model's weights
contain the instruction graph and program ROM; it stores PC, accumulator,
halt bit, and 16 RAM bytes in Gated DeltaNet state.

If Qwen calls the tool, the host parses a byte `x`, inserts it as the CPU's
initial accumulator token, and asks the CPU model to generate until HALT.
The host decodes the final accumulator and gives that value back to Qwen in
a tool response. The host does not perform the addition. A missing, malformed,
or out-of-range call is recorded as no valid tool call, with no CPU execution.

This is a **toy host-routed tool**, separate from the 9B checkpoint that
loads bytecode into recurrent ROM. The host handles tool-call parsing, token
routing, and response formatting. The Qwen text tower is not trained to
control the byte CPU and may choose to answer without invoking it. The CPU
model runs in a separate cache so its raw trace tokens do not affect Qwen's
text context. The demo handles at most one tool call per question.

## Run

From Qwenputer's pinned `experimental/riscv` environment with `hf` deps:

```sh
export PYTHONPATH=/path/to/qwenputer/experimental/riscv:/path/to/gdn-cpu-lab
python -m gdn_cpu.byte_bridge /path/to/Qwen3.5-9B \
  --ask 'Use byte_add7: what is 201 plus 7 modulo 256?' \
  --out /path/to/byte-bridge-result.json
```

The report exposes `tool_called`, the parsed `tool_input`, the CPU's
model-generated trace and result, and the final text. A result from this
command is evidence of autonomous tool selection only if `tool_called` is
true. The program is fixed in the CPU model; supporting arbitrary bytecode
from a user prompt requires a runtime ROM loader.

## Frozen tool-selection check

Eight prompts were frozen before any 9B bridge run in
[`eval/frozen/byte_bridge.jsonl`](eval/frozen/byte_bridge.jsonl), SHA-256
`e5747da0f76b120b82ab948bedae8015e74e6e1d10847d3c2544d196be6dec83`.
Four name the tool and four describe the byte operation without naming it.
They include inputs `249`, `250`, and `255`, which wrap around. The runner
loads both models once and appends one JSONL row per case, so an interrupted
run can resume:

```sh
python -m gdn_cpu.byte_bridge_eval /path/to/Qwen3.5-9B \
  --fixture /path/to/gdn-cpu-lab/eval/frozen/byte_bridge.jsonl \
  --out /path/to/gdn-cpu-lab/results/byte_bridge_9b.jsonl
```

The summary reports emitted tool calls, valid parsed calls, CPU errors,
correct tool inputs, exact CPU results, parsed final answers, and correct
boxed final answers separately. The evaluator
checks answers with the frozen gold values; that scoring arithmetic is not
used by the host bridge to execute a tool call.

The CPU execution path was checked locally on Apple M4 MPS with frozen
inputs `0`, `201`, `249`, and `255`. The generated results were respectively
`7`, `208`, `0`, and `6`, including wraparound. All four runs halted after
two instructions. This test did not involve 9B tool selection. The local
run used Transformers' reference Gated DeltaNet kernels. The exact generated
tokens and package versions are in [byte_bridge_mps.json](results/byte_bridge_mps.json).

On the real 9B CUDA run, all eight frozen prompts produced exact boxed final
answers. The 9B text model emitted a valid `byte_add7` call in seven cases:
three of four prompts that named the tool and all four that described the
operation without naming it. All seven parsed inputs were correct, and the
separate model-generated CPU traces returned the exact byte result with no
CPU errors. The one remaining prompt was answered correctly without a tool
call. See the [summary](results/byte_bridge_9b.summary.json) and
[per-case records](results/byte_bridge_9b.jsonl). This small test does not
show that the merged 9B text tower autonomously invokes its own byte CPU.
