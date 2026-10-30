# Held-out Qwen3.5-9B arithmetic comparison

The 24 expressions in [arithmetic.jsonl](eval/frozen/arithmetic.jsonl) were
frozen before the 9B run, with seed `20261030` and SHA-256
`a4277e1e79229b258319454b237f2f84749c67da3ad1d60f0dcbd3fc6f4e0896`.
The three earlier smoke expressions were excluded. Each arm used greedy
decoding and a 4,096 text-token limit. The host-tool and merged arms received
identical prompt token IDs for every case, verified by saved hashes. The base
arm received a by-hand arithmetic instruction. Strict scoring requires an
exact final answer in `\\boxed{}`; a last-number fallback is recorded only as
a diagnostic.

| Arm | Strict exact | Cases with tool call | Hit text limit | CPU tokens | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 9B, no tool | 4/24 | 0/24 | 17/24 | 0 | 0 |
| Base 9B with host calculator | 23/24 | 24/24 | 1/24 | 0 | 0 |
| Merged 9B with RV32 calculator | 23/24 | 24/24 | 1/24 | 160,008 | 0 |

The same `frac-2` case was the sole miss in both tool arms. The host arm made
three completed calls and the merged arm four; each then exhausted its text
limit while continuing its explanation. The calculator itself returned the
exact fraction. For the merged arm, all 24 full-stream replays were exact
across 178,623 positions, with zero head mismatches, lockstep errors, wrong
circuit predictions, or ownership mismatches. Replay includes prompts and
model text as well as CPU tokens. [summary.json](eval/results/summary.json)
and the [raw records](eval/results/) preserve every output and diagnostic.

This result demonstrates exact execution of the tested RV32 calculator calls
inside a pretrained 9B model and equal answer accuracy to the host tool on
this small frozen suite. It does not establish a general arithmetic advantage
over a host calculator. The merged arm's recorded 993 seconds includes an
independent full-stream replay after each case, whereas the host arm's 130
seconds does not; those totals are not comparable inference latencies. The
4,096-token limit also counts forced calculator response text in the merged
arm but only model-generated text in the host arm. Both capped cases are the
same one here, but the limit accounting prevents a strict budget-parity claim.

The runs used BF16 Qwen3.5-9B weights, Transformers 5.19.0, PyTorch 2.14.1
CUDA 13.0, and the reference PyTorch Gated DeltaNet kernels on one Runpod
RTX PRO 6000 Blackwell Server Edition GPU. The source model was pinned to
Hugging Face snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; the
Qwenputer checkout was pinned to
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. The 9B RV32 changes are
in [qwenputer-9b-port.patch](qwenputer-9b-port.patch). [eval/README.md](eval/README.md)
has the runner commands and scoring rules.
