"""Optional BF16 tensor check of the one-hot Gated DeltaNet RAM rule."""

from __future__ import annotations

import random


def main() -> None:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    state = torch.zeros((16, 8), dtype=torch.bfloat16, device=device)
    expected = [0] * 16
    rng = random.Random(1925)

    for _ in range(1_000):
        address = rng.randrange(16)
        value = rng.randrange(256)
        key = torch.zeros(16, dtype=torch.bfloat16, device=device)
        key[address] = 1
        bits = torch.tensor([(value >> i) & 1 for i in range(8)],
                            dtype=torch.bfloat16, device=device)
        old = key @ state
        state = state + torch.outer(key, bits - old)
        expected[address] = value

    result = [sum(int(state[row, bit].item()) << bit for bit in range(8))
              for row in range(16)]
    if result != expected:
        raise AssertionError(f"BF16 RAM mismatch: {result} != {expected}")
    print(f"BF16 addressed RAM: 1,000 writes exact on {device}")


if __name__ == "__main__":
    main()
