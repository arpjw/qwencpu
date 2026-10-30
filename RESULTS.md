# Completed experiment

This lab developed an independent 8-bit ISA and CPU circuit inside Qwen3.5,
using Qwenputer's generic Qwen3.5 compiler backend. Its CPU uses actual
Gated DeltaNet state for ROM, RAM, and registers. A separate patch ports the
published RV32 calculator merge from
Qwen3.5-4B to Qwen3.5-9B while preserving 9B's untied output head. The
[rebuild instructions](REPRODUCE.md) contain the pinned inputs and commands.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Dynamic byte CPU, stock Qwen3.5 | 12/12 frozen short programs and 2/2 high-address programs exact on MPS and CUDA | [DYNAMIC_ROM.md](DYNAMIC_ROM.md) |
| Pretrained 9B dynamic byte CPU | 124/124 short-suite tokens, 62 RAM snapshots; 18/18 high-address tokens, nine RAM snapshots; ROM copies exact | [BYTE9B.md](BYTE9B.md) |
| Optimized Gated DeltaNet | Same byte suites exact with Flash Linear Attention kernels on CUDA; causal convolution remained on reference kernels | [optimized logs](results/byte9b_dynamic_highbank_fla.log) |
| 9B text before CPU trace | 12/12 fixed prompts kept the same next token and first eight greedy tokens as base 9B | [text check](results/byte9b_dynamic_textcheck.json) |
| 9B text after in-band CPU trace | Next token changed on both tested chat prompts; compressed CPU position IDs did not fix this | [position check](results/byte9b_dynamic_positions.json) |
| Same-checkpoint text-to-byte-CPU bridge | 7/8 valid tool calls with exact generated CPU traces, 8/8 exact boxed answers, using separate caches | [bridge summary](results/byte9b_bridge_9b.summary.json) |
| Separate RV32 arithmetic comparison | Base 4/24; base with host calculator 23/24; merged RV32 calculator 23/24. All 24 merged replays exact | [BENCHMARK.md](BENCHMARK.md) |

All counts refer to small frozen or fixed suites, not general model
reliability. The same-checkpoint byte bridge still relies on host tool
parsing, raw-token routing, and response formatting. The RV32 benchmark and
the independent byte CPU test different circuits. No model weights, paid
resources, or deployment are included in this deliverable.

## Validation and spend

The final CPU and compiler-backed suite passed 18/18 tests in the reference
kernel environment; see [final_tests.log](results/final_tests.log). The
optimized Flash Linear Attention package works for the tested CUDA paths but
attempts to launch Triton on CPU-only synthetic tests, so those tests used
the reference environment. The separate Mac test process stalled while
loading Torch; its earlier compiler-backed run passed 18/18.

Two bounded Runpod sessions used temporary Pods and network volumes. The
second ran for about two hours on a 96 GB GPU quoted at $2.49/hour. All
project resources were deleted after the small artifacts were copied, and a
post-deletion inventory found none remaining. At the quoted rates, the two
sessions and volumes together are estimated at about **$6.30**. Provider
billing was still catching up at teardown, so this is not a final invoice.
