"""Exercise this lab's 8-bit state rows in an actual Qwen3.5 DeltaNet layer.

The probe builds a small, untrained Qwen3.5TextModel with hand-set BF16
weights. It writes and reads the CPU's 16 RAM bytes plus accumulator, program
counter, and halt flag. The remaining CPU logic is not in this model.

Run with ``python -m gdn_cpu.embedding_probe`` after installing PyTorch and
Transformers with Qwen3.5 support. No pretrained checkpoint is downloaded.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from gdn_cpu.reference import RAM_SIZE


N_ROWS = RAM_SIZE + 3
HIDDEN = 512
HEAD_DIM = 128
N_HEADS = 16
BALLAST = float(2**40)
FEATURE_SCALE = float(2**36)
GATE = 128.0
ROW_REF = float(2**21)

# Residual positions. The other 456 dimensions remain zero in this probe.
WRITE = 2
WRITE_ROW = 3
READ_ROW = WRITE_ROW + N_ROWS
DATA = READ_ROW + N_ROWS
RESULT = DATA + 8


@dataclass(frozen=True)
class Operation:
    """One token's requested state write and addressed read."""

    read: int | None
    write: int | None = None
    value: int = 0

    def __post_init__(self) -> None:
        if self.read is not None and not 0 <= self.read < N_ROWS:
            raise ValueError("read row out of range")
        if self.read is None and self.write is not None:
            raise ValueError("text tokens cannot write state")
        if self.write is not None and not 0 <= self.write < N_ROWS:
            raise ValueError("write row out of range")
        if not 0 <= self.value <= 255:
            raise ValueError("value must be a byte")


def make_model(dtype=None):
    """Return a real Qwen3.5 module with a hard-coded one-head byte store."""
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    if dtype is None:
        dtype = torch.bfloat16
    config = Qwen3_5TextConfig(
        vocab_size=16,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=N_HEADS,
        linear_num_value_heads=N_HEADS,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        tie_word_embeddings=False,
    )
    model = Qwen3_5TextModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        layer = model.layers[0]
        attn = layer.linear_attn
        qkv = attn.in_proj_qkv.weight
        z = attn.in_proj_z.weight
        beta = attn.in_proj_b.weight
        out = attn.out_proj.weight

        # Two ballast dimensions pin Qwen's input RMSNorm divisor at 2^36.
        # A residual feature x then arrives at a projection as x / 2^36.
        def project(target, terms):
            for dim, coefficient in terms.items():
                target[dim] = coefficient / 16 if dim == 0 else coefficient * FEATURE_SCALE

        q_base, k_base, v_base = 0, N_HEADS * HEAD_DIM, 2 * N_HEADS * HEAD_DIM
        for row in range(N_ROWS):
            project(qkv[q_base + row], {READ_ROW + row: GATE})
            project(qkv[k_base + row], {WRITE_ROW + row: GATE, WRITE: GATE, 0: -GATE})
        project(qkv[v_base], {0: ROW_REF})
        project(qkv[v_base + 1], {0: ROW_REF})
        for bit in range(8):
            project(qkv[v_base + 2 + bit], {DATA + bit: GATE})
            out[RESULT + bit, 2 + bit] = 16.0
        project(beta[0], {0: GATE})  # sigmoid(128) rounds to one
        for bit in range(8):
            project(z[2 + bit], {0: GATE})
        attn.dt_bias.fill_(-110.0)  # a=0 gives no decay
        attn.A_log.zero_()
        attn.norm.weight.fill_(1.0)
        attn.conv1d.weight[:, 0, -1] = 1.0
    return model.to(dtype).eval()


def encode(operations):
    """Make direct input embeddings for addressed byte operations."""
    import torch

    x = torch.zeros(len(operations), HIDDEN, dtype=torch.float32)
    for i, op in enumerate(operations):
        if op.read is None:
            # Stand-in for an ordinary text residual: no chip features, but
            # nonzero dimensions outside the circuit's projection support.
            x[i, 400] = 1000
            continue
        x[i, :2] = BALLAST
        x[i, READ_ROW + op.read] = 1
        if op.write is not None:
            x[i, WRITE] = 1
            x[i, WRITE_ROW + op.write] = 1
            for bit in range(8):
                x[i, DATA + bit] = (op.value >> bit) & 1
    return x


def oracle(operations):
    """Independent byte dictionary; same-token reads see the latest write."""
    rows = [0] * N_ROWS
    observed = []
    for op in operations:
        if op.write is not None:
            rows[op.write] = op.value
        observed.append(rows[op.read] if op.read is not None else 0)
    return observed


def run_probe(operations, *, dtype=None, decode_split=None, device="cpu"):
    """Return the actual layer's decoded read values and max residual error."""
    import torch

    if not operations:
        raise ValueError("operations must be nonempty")
    model = make_model(dtype).to(device)
    x = encode(operations)
    emitted = []

    def capture(_module, _inputs, output):
        emitted.append(output[0] if isinstance(output, tuple) else output)

    hook = model.layers[0].register_forward_hook(capture)
    try:
        with torch.no_grad():
            if decode_split is None:
                model(inputs_embeds=x[None].to(device=device, dtype=next(model.parameters()).dtype),
                      use_cache=False)
            else:
                if not 1 <= decode_split <= len(operations):
                    raise ValueError("decode_split out of range")
                first = model(inputs_embeds=x[None, :decode_split].to(
                                  device=device, dtype=next(model.parameters()).dtype),
                              use_cache=True)
                cache = first.past_key_values
                for t in range(decode_split, len(operations)):
                    result = model(inputs_embeds=x[None, t:t + 1].to(
                                       device=device, dtype=next(model.parameters()).dtype),
                                   past_key_values=cache, use_cache=True)
                    cache = result.past_key_values
    finally:
        hook.remove()
    y = torch.cat([part[0] for part in emitted], dim=0).float()[:, RESULT:RESULT + 8]
    wanted = oracle(operations)
    expected = torch.tensor([[(byte >> bit) & 1 for bit in range(8)] for byte in wanted],
                            dtype=torch.float32, device=device)
    error = float((y - expected).abs().max())
    decoded = [sum(int(round(float(bit))) << i for i, bit in enumerate(row)) for row in y]
    return decoded, error


def sample_operations(seed=0, count=128):
    """Stress overwrites and reads across all CPU state rows."""
    rng = random.Random(seed)
    operations = [Operation(read=row, write=row, value=rng.randrange(256))
                  for row in range(N_ROWS)]
    for _ in range(count):
        if rng.randrange(11) == 0:
            operations.append(Operation(read=None))
        else:
            operations.append(Operation(read=rng.randrange(N_ROWS),
                                        write=rng.randrange(N_ROWS) if rng.randrange(3) else None,
                                        value=rng.randrange(256)))
    return operations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--prefill-only", action="store_true")
    args = parser.parse_args()
    operations = sample_operations(args.seed, args.count)
    expected = oracle(operations)
    modes = (("prefill", None),) if args.prefill_only else (("prefill", None), ("decode", 17))
    for mode, split in modes:
        found, error = run_probe(operations, decode_split=split, device=args.device)
        print(f"{mode}: {len(found)} reads, {sum(a != b for a, b in zip(found, expected))} "
              f"byte mismatches, max bit error {error:g}")
        if found != expected or error != 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
