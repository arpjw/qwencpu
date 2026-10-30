"""Generate deterministic arithmetic and 8-bit program cases before inference.

The arithmetic oracle uses Python's integer and Fraction operations. The program
oracle is the separately implemented gdn_cpu.reference interpreter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path

from gdn_cpu.reference import CPUState, step


SEED = 20261030


def arithmetic_cases(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    for i in range(4):
        a = rng.randrange(10**12, 10**17)
        b = rng.randrange(10**11, 10**16)
        cases.append({"id": f"add-{i}", "kind": "add", "expression": f"{a}+{b}",
                      "gold": str(a + b)})
    for i in range(4):
        a = rng.randrange(10**8, 10**12)
        b = rng.randrange(10**7, 10**11)
        cases.append({"id": f"mul-{i}", "kind": "mul", "expression": f"{a}*{b}",
                      "gold": str(a * b)})
    for i in range(4):
        factor = rng.randrange(10**3, 10**6)
        a = factor * rng.randrange(10**5, 10**9)
        b = factor * rng.randrange(10**4, 10**8)
        cases.append({"id": f"gcd-{i}", "kind": "gcd", "expression": f"gcd({a},{b})",
                      "gold": str(math.gcd(a, b))})
    for i in range(4):
        numerator = rng.randrange(10**9, 10**14)
        denominator = rng.randrange(10**8, 10**12)
        value = Fraction(numerator, denominator)
        gold = (str(value.numerator) if value.denominator == 1
                else f"{value.numerator}/{value.denominator}")
        cases.append({"id": f"frac-{i}", "kind": "frac",
                      "expression": f"{numerator}/{denominator}", "gold": gold})
    for i in range(4):
        a = rng.randrange(10**12, 10**17)
        b = rng.randrange(10**11, 10**16)
        cases.append({"id": f"sub-{i}", "kind": "sub", "expression": f"{b}-{a}",
                      "gold": str(b - a)})
    for i in range(4):
        a = rng.randrange(10**12, 10**16)
        b = rng.randrange(10**5, 10**8)
        if i % 2 == 0:
            expression, gold = f"{a}//{b}", a // b
            kind = "floor_div"
        else:
            expression, gold = f"{a}%{b}", a % b
            kind = "mod"
        cases.append({"id": f"divmod-{i}", "kind": kind, "expression": expression,
                      "gold": str(gold)})
    excluded = {"123456789*987654321", "gcd(84,18)", "84/18"}
    if any(case["expression"] in excluded for case in cases):
        raise AssertionError("held-out suite overlaps a prior smoke expression")
    return cases


def _state(state: CPUState) -> dict:
    return {"pc": state.pc, "acc": state.acc, "ram_hex": state.ram.hex(),
            "halted": state.halted}


def _trace(program: bytes, max_steps: int = 64) -> list[dict]:
    state = CPUState()
    trace = [_state(state)]
    for _ in range(max_steps):
        state = step(state, program)
        trace.append(_state(state))
        if state.halted:
            return trace
    raise ValueError("generated program did not halt")


def program_cases(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    programs: list[tuple[str, bytes]] = []
    for i in range(8):
        initial = rng.randrange(256)
        increment = rng.randrange(256)
        address = rng.randrange(16)
        # Store, overwrite accumulator, then load and add. Includes wraparound.
        code = bytes([0x10, initial, 0x30, address, 0x10, 0,
                      0x40, address, 0x20, increment, 0xFF])
        programs.append((f"ram-add-{i}", code))
    programs.extend([
        ("branch-taken", bytes([0x10, 1, 0x50, 6, 0x10, 99, 0x20, 255, 0xFF])),
        ("branch-not-taken", bytes([0x10, 0, 0x50, 6, 0x20, 7, 0xFF])),
        ("nop-halt", bytes([0x00, 0x00, 0xFF])),
        ("wraparound", bytes([0x10, 255, 0x20, 1, 0xFF])),
    ])
    return [{"id": name, "program_hex": code.hex(), "max_steps": 64,
             "expected_trace": _trace(code)} for name, code in programs]


def write_cases(output_dir: Path, seed: int = SEED) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = {"arithmetic.jsonl": arithmetic_cases(seed),
              "programs.jsonl": program_cases(seed)}
    for name, rows in suites.items():
        (output_dir / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        json.dumps({"seed": seed, "arithmetic_cases": len(suites["arithmetic.jsonl"]),
                    "program_cases": len(suites["programs.jsonl"])}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    write_cases(args.out, args.seed)


if __name__ == "__main__":
    main()
