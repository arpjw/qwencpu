"""Run frozen byte programs through one shared dynamic-ROM Qwen checkpoint.

This evaluator uses expected traces from a frozen JSONL file. It does not
invoke the reference emulator. The same hand-constructed model object and
weights are reused across all programs; each generate call starts fresh cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import A_BIT, A_REF, Head
from gdn_cpu.qwen8_dynamic import PAD, TICK, build_standalone, program_tokens
from gdn_cpu.qwen8_eval import ram_snapshot


def _state_token(state: dict) -> int:
    return int(state["pc"]) | (int(state["acc"]) << 8) | (int(state["halted"]) << 16)


def _head_matrix(cache, circuit, name: str):
    node = next(n for n in circuit.nodes if isinstance(n, Head) and n.name == name)
    layer = (node.level - 1) // 2
    index = sum(isinstance(n, Head) and n.level == node.level
                for n in circuit.nodes[:circuit.nodes.index(node)])
    value_heads_per_key = circuit.target.value_heads // circuit.target.key_heads
    return cache.layers[layer].recurrent_states[0][0, index * value_heads_per_key].float()


def rom_snapshot(cache, circuit, program_length: int) -> dict[str, str]:
    """Decode both independent ROM copies from actual Transformers cache."""
    out = {}
    for name in ("op", "arg"):
        matrices = [_head_matrix(cache, circuit, f"rom.{name}.{bank}")
                    for bank in range((circuit.rom_size + 127) // 128)]
        values = bytearray(program_length)
        for address in range(program_length):
            row = matrices[address // 128][address % 128]
            if tuple(row[:2].tolist()) != (A_REF, A_REF):
                raise ValueError(f"{name} ROM ballast wrong at {address}")
            for bit in range(8):
                value = float(row[2 + bit])
                if value not in (0.0, A_BIT):
                    raise ValueError(f"{name} ROM bit wrong at {address}, bit {bit}: {value}")
                values[address] |= int(value == A_BIT) << bit
        out[name] = values.hex()
    return out


@torch.no_grad()
def evaluate_case(model, circuit, case: dict, *, rom_size: int, device: str) -> dict:
    start = time.monotonic()
    program = bytes.fromhex(case["program_hex"])
    trace = case["expected_trace"]
    expected = [token for state in trace[1:] for token in (TICK, _state_token(state))]
    prefix = [PAD, *program_tokens(program, rom_size=rom_size), _state_token(trace[0])]
    prompt = torch.tensor([prefix], device=device)
    generated = model.generate(prompt, max_new_tokens=len(expected), do_sample=False,
                               pad_token_id=PAD, eos_token_id=None,
                               cache_implementation="dynamic")[0].tolist()[len(prefix):]
    observed_ram = []
    ram_error = None
    observed_rom = {}
    rom_error = None
    try:
        output = model(input_ids=prompt, use_cache=True)
        cache = output.past_key_values
        try:
            observed_rom = rom_snapshot(cache, circuit, len(program))
        except (RuntimeError, ValueError) as error:
            rom_error = f"{type(error).__name__}: {error}"[:400]
        for i in range(len(trace) - 1):
            tick = torch.tensor([[generated[2 * i]]], device=device)
            output = model(input_ids=tick, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            observed_ram.append(ram_snapshot(cache, circuit).hex())
            if i + 1 < len(trace) - 1:
                state = torch.tensor([[generated[2 * i + 1]]], device=device)
                output = model(input_ids=state, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
    except (IndexError, RuntimeError, ValueError) as error:
        ram_error = f"{type(error).__name__}: {error}"[:400]
    expected_ram = [state["ram_hex"] for state in trace[1:]]
    return {"id": case["id"], "program_hex": program.hex(),
            "steps": len(trace) - 1, "tokens": len(expected),
            "token_matches": sum(a == b for a, b in zip(generated, expected)),
            "tokens_exact": generated == expected,
            "ram_exact": observed_ram == expected_ram and ram_error is None,
            "rom_exact": observed_rom == {"op": program.hex(), "arg": program.hex()}
            and rom_error is None,
            "observed_rom": observed_rom, "rom_error": rom_error,
            "expected_ram_after_ticks": expected_ram,
            "observed_ram_after_ticks": observed_ram, "ram_error": ram_error,
            "generated": generated, "expected": expected,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--rom-size", type=int, default=16)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not cases or any(len(bytes.fromhex(case["program_hex"])) > args.rom_size for case in cases):
        raise ValueError("all programs must fit the selected ROM size")
    model, circuit = build_standalone(rom_size=args.rom_size)
    model.to(args.device)
    report = {"cases_sha256": hashlib.sha256(raw).hexdigest(),
              "model_builds": 1, "rom_size": args.rom_size,
              "vocab": circuit.vocab, "circuit": circuit.stats(),
              "hidden": model.config.hidden_size, "layers": model.config.num_hidden_layers,
              "device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "cases": []}
    for case in cases:
        result = evaluate_case(model, circuit, case, rom_size=args.rom_size,
                               device=args.device)
        report["cases"].append(result)
        fields = ("id", "tokens_exact", "rom_exact", "ram_exact", "tokens", "seconds")
        print(json.dumps({key: result[key] for key in fields}), flush=True)
    report["all_exact"] = all(c["tokens_exact"] and c["rom_exact"] and c["ram_exact"]
                              for c in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "cases": len(cases),
                      "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
