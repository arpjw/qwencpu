"""Score the recurrent Qwen byte CPU against a frozen fixture.

This adapter imports no CPU reference implementation. It builds the compiled
model, runs stock greedy generation, and compares tokens and final GDN RAM to
the fixture produced separately by work/generate_byte12.py.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import A_BIT, A_REF, Head
from gdn_cpu.qwen8_recurrent import PAD, build_standalone


def ram_snapshot(cache, circuit) -> bytes:
    ram_node = next(n for n in circuit.nodes if n.name == "ram")
    layer = (ram_node.level - 1) // 2
    head = sum(isinstance(n, Head) and n.level == ram_node.level
               for n in circuit.nodes[:circuit.nodes.index(ram_node)])
    matrix = cache.layers[layer].recurrent_states[0][0, head].float()
    ram = bytearray(16)
    for address in range(16):
        row = matrix[address]
        if tuple(row[:2].tolist()) not in ((0.0, 0.0), (A_REF, A_REF)):
            raise ValueError(f"invalid RAM ballast at address {address}")
        for bit in range(8):
            value = float(row[2 + bit])
            if value not in (0.0, A_BIT):
                raise ValueError(f"invalid RAM bit at address {address}, bit {bit}: {value}")
            ram[address] |= int(value == A_BIT) << bit
    return bytes(ram)


@torch.no_grad()
def evaluate_case(case: dict, device: str) -> dict:
    start = time.monotonic()
    model, circuit = build_standalone(bytes.fromhex(case["program"]))
    model.to(device)
    prompt = torch.tensor([[PAD, 0]], device=device)
    generated = model.generate(prompt, max_new_tokens=len(case["tokens"]), do_sample=False,
                               pad_token_id=PAD, eos_token_id=None,
                               cache_implementation="dynamic")[0].tolist()[2:]
    matches = sum(a == b for a, b in zip(generated, case["tokens"], strict=False))

    # The final state token is predicted after the final TICK. Feed the prompt
    # and every generated token before it, then read the actual cache's RAM.
    replay = torch.tensor([[PAD, 0, *generated[:-1]]], device=device)
    output = model(replay, use_cache=True)
    ram = ram_snapshot(output.past_key_values, circuit)
    result = {"name": case["name"], "steps": case["steps"],
              "tokens": len(case["tokens"]), "token_matches": matches,
              "tokens_exact": generated == case["tokens"],
              "ram_exact": ram.hex() == case["ram"],
              "expected_ram": case["ram"], "actual_ram": ram.hex(),
              "generated": generated,
              "seconds": round(time.monotonic() - start, 3)}
    del output, model, circuit
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", action="append", help="run only the selected named case")
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    if fixture["schema"] != 1:
        raise ValueError("unsupported fixture schema")
    cases = [c for c in fixture["cases"] if not args.name or c["name"] in args.name]
    report = {"fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
              "device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "cases": []}
    for case in cases:
        result = evaluate_case(case, args.device)
        report["cases"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "generated"}), flush=True)
    report["all_exact"] = all(c["tokens_exact"] and c["ram_exact"] for c in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_exact": report["all_exact"], "n_cases": len(cases),
                      "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
