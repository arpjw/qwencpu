# Model port feasibility

This was the source-inspection plan before the model port. The independent
8-bit CPU proof was developed without pretrained weights. The separate 9B
merge was later validated on a bounded Runpod session; see [VALIDATION.md](VALIDATION.md).

## Architecture check (October 29, 2026)

| Model | Text layers | Gated DeltaNet key/value heads | Text hidden size | Published RV32 layout |
| --- | ---: | ---: | ---: | --- |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) | 32 | 16 / 32 | 2560 | Implemented by Qwenputer |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json) | 32 | 16 / 32 | 4096 | Key/value head ratio fits; later port tested |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) | 24 | 16 / 16 | 2048 | Needs a different memory layout or more sublayers |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json) | 24 | 16 / 16 | 1024 | Needs a different memory layout or more sublayers |

All four text configs place a full-attention layer after each three Gated DeltaNet layers. Qwenputer's [core](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/chipformer/rv/core.py) explicitly packs two value heads per key head into its ROM, RAM, and bridge tables. Its [design notes](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md) say 0.8B was skipped because it has only one value head per key head. The same head-ratio issue applies to 2B. A smaller custom CPU can use one value head per key head if its memory map is designed for it.

## 9B-specific obstacle

The 4B config ties the output head to the token embedding. The 9B config sets `tie_word_embeddings: false`, and its [checkpoint index](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/model.safetensors.index.json) lists a separate `lm_head.weight`. The published [RV merge](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/chipformer/rv/merge.py) copies only `model.language_model.*` tensors and forces `tie_word_embeddings=True`; a direct run on 9B would therefore discard its learned output head and cannot establish base-model fidelity. A correct 9B port must preserve and widen the separate output head, add circuit token logits to it, and test that ordinary text logits remain close to the base model. That change is more than a model-name swap.

The 9B source checkpoint index reports about 19.3 GB of weight data, before added layers, memory for decoding, and construction buffers. The 4B source index reports about 9.3 GB. These are file sizes, not measured GPU requirements or costs. A 9B merge should not assume the 24 GB GPU used for the published 4B checkpoint is sufficient.

## Minimum path

1. Finish an independent byte-addressed CPU and reference emulator. Verify each state transition and long free-running traces in exact arithmetic.
2. Compile only the memory primitive and tiny instruction set into real Qwen3.5 Gated DeltaNet and SwiGLU modules. Verify BF16 writes, reads, and state persistence through the actual model kernels.
3. Integrate with 4B first to validate layer insertion, token routing, checkpoint construction, and text fidelity against a known architecture.
4. Port to 9B with its untied output head intact. Check tensor shapes and tokenizer IDs before merging; then test CPU traces, ordinary text logits, and end-to-end tool-call selection against a reference interpreter.
5. Benchmark on held-out tasks against the untouched base model and the same model with a host tool. Report CPU correctness separately from the language model's decision to invoke it.

The published [Qwenputer code](https://github.com/divagr18/qwenputer) and [write-up](https://divagr.com/blog/qwenputer) are useful design references. This page records the initial architecture assessment, not the later result. See [BYTE9B.md](BYTE9B.md) for the independent byte CPU merge and [BENCHMARK.md](BENCHMARK.md) for the separate RV32 arithmetic comparison.
