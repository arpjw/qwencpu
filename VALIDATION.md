# Qwen3.5-9B port validation

This records a bounded Runpod validation on October 29, 2026. The port is a
patch to the published Qwenputer RV32 merge, pinned to upstream commit
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. The base model was
`Qwen/Qwen3.5-9B` snapshot
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. No model weights are included
in this deliverable.

## Environment and build

- Runpod Secure Cloud, one NVIDIA RTX PRO 6000 Blackwell Server Edition
  (96 GB VRAM), with a 150 GB temporary standard network volume.
- Python 3.12.3, PyTorch 2.14.1+cu130, Transformers 5.19.0.
- Five focused 9B merge tests passed. They check the untied output head,
  config, and synthetic text/circuit logits.
- Circuit allocation produced 11 added layers, 1,265 slots, and hidden size
  5,632. The merged text-only checkpoint contained 43 layers and occupied
  about 35 GB on the remote volume. The merge exited successfully.

The base model and merge ran with the reference causal-convolution and
Gated DeltaNet kernels because the optional optimized kernels were absent.

## End-to-end checks

`RvChat` used the real merged 9B checkpoint, the `calc-bridge.elf` firmware,
head agreement checks, and reference-machine replay:

| Prompt | Model result | Circuit-owned positions | Head mismatches | Replay errors |
| --- | --- | ---: | ---: | ---: |
| `123456789 * 987654321` | `121932631112635269` | 2,460 | 0 | 0 |
| `gcd(84,18)` | `6` | 2,663 | 0 | 0 |
| `84/18` | `14/3` | 3,152 | 0 | 0 |

The [text fidelity result](results/qwen-textcheck.json) compared base and
merged models on the published script's 16 prompts, using 32 generated tokens
per prompt. Mean KL(base || merged) was **0.000483**, versus **0.000402** for
base-model SDPA versus eager kernels. Greedy continuations were identical on
**13/16** prompts, and tokenizer encodings matched on **4/4** samples.

These checks support a working 9B port for the tested calculator calls. They
do not establish reliability across arbitrary programs, conversations, or
benchmarks. The tokenizer emitted a generic regex warning on the merged model;
the four sampled encodings matched, but a broader tokenizer audit was not run.
A later [24-case arithmetic comparison](BENCHMARK.md) used this patched
checkpoint and is reported separately from this initial validation session.

## Reproduce

Copy [qwenputer-9b-port.patch](qwenputer-9b-port.patch) into a workspace, then
apply it to the pinned upstream checkout on a sufficiently large CUDA machine:

```sh
git clone https://github.com/divagr18/qwenputer.git
cd qwenputer
git checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c
git apply /path/to/qwenputer-9b-port.patch
cd experimental/riscv
uv sync --extra hf --no-dev
uv run --extra hf --no-dev hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir qwen35-9b
uv run --extra hf --no-dev python -m chipformer.rv.merge qwen35-9b --out qwen35-9b-merged
uv run --extra hf --no-dev python -m chipformer.rv.chat qwen35-9b-merged \
  --firmware tests/data/rv/calc-bridge.elf \
  --ask 'Use calc: what is 123456789 * 987654321?' --check
uv run --extra hf --no-dev python -m chipformer.rv.textcheck \
  qwen35-9b qwen35-9b-merged --out textcheck.json
```

The source patch, [merged config](results/config.json),
[circuit metadata](results/chipformer.json), and [run logs](results/) are saved
here. The merged checkpoint itself was deleted with the temporary volume.

## Cost and teardown

The project Pod ran from approximately 00:21:41 to 00:49:34 UTC on October 30,
2026 (October 29 in the user's Pacific time zone). At the quoted $2.49/hour
GPU rate plus temporary storage, estimated cost is **about $1.17**, well under
the authorized $50 cap. The Runpod billing endpoint had not posted this Pod's
charge at teardown, so $1.17 is an estimate, not verified billing.

The temporary Pod and volume were deleted. A post-deletion inventory found
neither project resource.
