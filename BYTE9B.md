# Byte CPU inside Qwen3.5-9B

`gdn_cpu.byte9b_merge` places this lab's independent 8-bit ISA circuit before
the Qwen3.5-9B text tower. It uses Qwenputer's MIT-licensed generic Qwen3.5
IR, gate library, weight emitter, and widening helpers as a compiler backend.
The source checkpoint's learned, untied `lm_head.weight` is copied exactly in
its original text columns. The added CPU rows use new output-code dimensions.

The recommended `dynamic` mode accepts program bytes through `LOAD` tokens.
Four Gated DeltaNet heads hold two copies of the 256-byte ROM; two more hold
PC, accumulator, halt bit, and 16 RAM bytes. The host supplies `LOAD` tokens
and the initial state token, then generation alternates a model-generated
`TICK` and model-generated state token. One `TICK` executes one instruction.
The same checkpoint can run many programs on fresh caches. No host emulator
participates during generation; it checks the trace afterward. The earlier
`recurrent` mode fixes its program in MLP weights, while `state` mode also
carries registers in generated state tokens.

## Build and check

Use the pinned Qwenputer checkout described in [VALIDATION.md](VALIDATION.md)
with its `hf` dependencies installed. Keep the base weights and output on a
machine with enough RAM and storage. One full-ROM checkpoint can run both
frozen program suites:

```sh
export PYTHONPATH=/path/to/qwenputer/experimental/riscv:/path/to/gdn-cpu-lab
python -m gdn_cpu.byte9b_merge /path/to/Qwen3.5-9B \
  --out /path/to/byte9b --mode dynamic --rom-size 256
python -m gdn_cpu.byte9b_check /path/to/byte9b \
  --fixture /path/to/gdn-cpu-lab/eval/frozen/programs.jsonl \
  --out /path/to/programs-check.json
python -m gdn_cpu.byte9b_check /path/to/byte9b \
  --fixture /path/to/gdn-cpu-lab/eval/frozen/dynamic256.jsonl \
  --out /path/to/high-bank-check.json
python -m gdn_cpu.byte9b_textcheck /path/to/Qwen3.5-9B /path/to/byte9b \
  --out /path/to/byte9b-textcheck.json
python -m gdn_cpu.byte9b_bridge_eval /path/to/byte9b \
  --fixture /path/to/gdn-cpu-lab/eval/frozen/byte_bridge.jsonl \
  --out /path/to/byte9b-bridge8.jsonl
```

The merge writes a stock text-only `Qwen3_5ForCausalLM` checkpoint and
`byte9b.json` metadata. `LOAD`, state, and `TICK` are raw model IDs beginning
at `chip_token_offset`. The copied tokenizer handles ordinary Qwen text only.
The checker supplies `PAD`, `LOAD(address, byte)` for each program byte, and
the initial CPU state. It compares every generated token with the independent
reference trace. It reads the actual Transformers cache to check both ROM
copies after loading and all 16 RAM bytes after each `TICK`. For the first
case in a fixture it also reports text-logit and top-token changes when a
text token follows the CPU trace. This handoff check measures the effect of
CPU positions on Qwen's context; it does not assume those positions are
invisible.

The text check uses 12 fixed, diverse chat prompts. It compares the base and
merged model's next-token distributions over the original text vocabulary,
top token, and up to eight greedy continuation tokens. It loads the models
sequentially to limit GPU memory use. For two prompts it also inserts an
actual model-generated CPU trace before a text token and compares that
continuation with the merged model's plain text path. The reported KL is one
next-token position per prompt, not a broad language benchmark. CPU tokens
still occupy positions in the stock decoder, so the handoff measurement may
show context changes even when the CPU and text weights are isolated. A
controlled second pass gives every CPU token the preceding text position and
keeps the next text token at its original position. The report compares the
ordinary and compressed-position paths, and checks CPU next-token logits
under teacher forcing. This only diagnoses positional effects; it does not
provide a free-running decoder that skips CPU positions.

## Same-checkpoint tool bridge

`gdn_cpu.byte9b_bridge` loads one dynamic 9B checkpoint. Its copied text
weights receive the `byte_add7` tool schema and choose whether to emit a
call. For a valid call, the host encodes the requested byte as the initial
accumulator, loads bytecode `20 07 ff` with raw `PAD` and `LOAD` token IDs,
and generates two `TICK` and state pairs on a fresh cache of the same model.
The host decodes the final model-generated accumulator and puts that value in
a normal text tool response. The final text continuation uses a fresh cache
containing text and the tool response, with no raw CPU token IDs. This avoids
the in-band positional effects measured by `byte9b_textcheck`. The host still
performs tool routing and validates the call; the model supplies the tool
selection and arithmetic result.

The eight fixed requests in `eval/frozen/byte_bridge.jsonl` are the same
explicit and implicit prompts used for the earlier two-model bridge. The
evaluator records whether a tool call was emitted, whether `byte_add7` and
its argument were parsed correctly, whether all four CPU tokens match the
independent emulator, and whether the final assistant answer is exact. It
scores the assistant continuation separately from the tool response, so the
tool value alone cannot count as a correct final answer. Output is resumable
JSONL with an adjacent summary file. These eight prompts test the bridge
protocol, not general tool-use reliability.

## Validation boundary

On a small random-weight, untied-head Qwen3.5 fixture, the full-256-byte
dynamic merge reused one checkpoint for both high-bank cases, including a
branch across address 128 and a halt at address 255. All 18 generated tokens,
both ROM copies, and nine RAM states were exact. A separate random fixture
showed that text-head columns were preserved bit for bit, CPU logits were
suppressed on text positions, and text logits were zero on CPU positions even
after a text prefix. These are shape and routing tests, not evidence about
pretrained 9B text quality.

The real Qwen3.5-9B dynamic checkpoint passed all 12 frozen short programs:
124/124 generated tokens, 62/62 RAM snapshots, and both loaded ROM copies
were exact. It also passed both high-address programs: 18/18 tokens, nine RAM
snapshots, and both ROM copies. The same suites passed with Flash Linear
Attention's optimized Gated DeltaNet kernels; causal convolution still used
Transformers' reference path. See the [standard suite](results/byte9b_dynamic_programs.json),
[high-address suite](results/byte9b_dynamic_highbank.json), and
[optimized results](results/byte9b_dynamic_programs_fla.json).

On 12 fixed chat prompts, base and merged 9B chose the same next token and
produced the same first eight greedy tokens in all 12 cases. Mean next-token
KL from base to merged was 0.000616. When a generated CPU trace was inserted
inside two chat prompts, both next-token choices changed, with KL 0.723 and
0.289 against the plain merged path. Compressing CPU position IDs reduced the
KL only to 0.703 and 0.214; both top choices still changed, although all 16
teacher-forced CPU predictions remained exact. See the [text check](results/byte9b_dynamic_textcheck.json)
and [position diagnostic](results/byte9b_dynamic_positions.json). This is a
material in-band handoff limitation.

The same-checkpoint bridge ran the eight frozen requests with one 9B model
object. It emitted seven valid calls, generated seven exact four-token CPU
traces, and gave eight exact boxed final answers. One prompt was answered
directly. See the [summary](results/byte9b_bridge_9b.summary.json) and
[per-case records](results/byte9b_bridge_9b.jsonl). The host still parses
the call, routes raw tokens, and formats the response. No checkpoint weights
are distributed with this lab.

Only valid, halting programs are in scope. The dynamic byte circuit and the
published RV32 calculator port are separate designs. These bounded tests do
not establish general language quality or autonomous execution of arbitrary
programs from natural language.
