# Qwen3.5-9B merge patch status

The patch in [`qwenputer-9b-port.patch`](qwenputer-9b-port.patch) applies to
[`divagr18/qwenputer`](https://github.com/divagr18/qwenputer) commit
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. It adapts the published RV32
merge, not the standalone 8-bit CPU proof in this directory.

Qwen3.5-9B's official `config.json` sets `tie_word_embeddings: false` and its
checkpoint index contains `lm_head.weight`. The published 4B merge forces a
tied output head and skips that tensor. This patch preserves the original
9B output-head columns for text tokens and takes the compiled circuit's output
columns from the newly generated embedding rows. It also retains the untied
setting in the merged text-only config. The source config fixture in the patch
was downloaded from
[`Qwen/Qwen3.5-9B/config.json`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
on October 29, 2026.

## Validation completed on October 29, 2026

- `git apply --check` passed against the pinned upstream checkout.
- The source files parse under Python 3.15.
- All five focused config and tensor/logit tests passed on Runpod with PyTorch
  2.14.1 and Transformers 5.19.0.
- The 9B checkpoint merged successfully into an approximately 35 GB, 43-layer
  text-only checkpoint. The source snapshot was
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Three end-to-end `calc` runs returned correct multiplication, GCD, and exact
  rational results. Each had zero head mismatches and zero replay errors.
- Text fidelity: mean KL 0.000483 versus a 0.000402 kernel noise floor; 13/16
  greedy continuations and 4/4 tokenizer samples matched.

See [VALIDATION.md](VALIDATION.md) and the [raw results](results/) for exact
commands, scope, and limitations. The merged checkpoint was removed with the
temporary Runpod volume after validation; the patch and logs are retained.

This is a port of the published RV32 model's merge logic. It is separate from
the independent 8-bit circuit proof in this directory. A later
[24-case held-out arithmetic comparison](BENCHMARK.md) completed with this
patch; that small synthetic result does not establish broad reliability.
