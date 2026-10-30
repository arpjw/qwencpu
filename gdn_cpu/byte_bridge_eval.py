"""Resumable 8-case evaluation of 9B tool selection and the byte CPU bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("runtime_error") is None]
    return {"cases": len(rows), "completed": len(completed),
            "tool_calls_emitted": sum(bool(row.get("tool_call_emitted")) for row in completed),
            "valid_tool_calls": sum(bool(row.get("tool_called")) for row in completed),
            "cpu_errors": sum(bool(row.get("cpu_error")) for row in completed),
            "correct_tool_inputs": sum(bool(row.get("tool_input_exact")) for row in completed),
            "correct_cpu_results": sum(bool(row.get("cpu_result_exact")) for row in completed),
            "cpu_consistent_with_requested_input": sum(
                bool(row.get("cpu_for_requested_input_exact")) for row in completed),
            "exact_parsed_answers": sum(bool(row.get("final_exact")) for row in completed),
            "exact_boxed_final_answers": sum(bool(row.get("final_boxed_exact"))
                                             for row in completed)}


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35.arith_bench import exact_fields
    from gdn_cpu.byte_bridge import PROGRAM, ask
    from gdn_cpu.qwen8_recurrent import build_standalone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_model", type=Path, help="base Qwen3.5-9B snapshot")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    cases = [json.loads(line) for line in fixture_bytes.splitlines() if line.strip()]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("fixture IDs are not unique")
    if args.limit is not None:
        cases = cases[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(args.out) if args.out.exists() else []
    if any(row.get("fixture_sha256") != fixture_sha for row in prior):
        raise ValueError("output contains results from a different fixture")
    done = {row["id"] for row in prior}

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModelForImageTextToText.from_pretrained(
        args.text_model, dtype=torch.bfloat16).to(args.device).eval()
    cpu_model, circuit = build_standalone(PROGRAM)
    cpu_model.to(args.device).eval()
    for case in cases:
        if case["id"] in done:
            continue
        try:
            response = ask(text_model, cpu_model, tokenizer, case["prompt"],
                           device=args.device, max_text_tokens=args.max_text_tokens)
            cpu = response["cpu"]
            final = exact_fields(case, response["final_text"])
            requested = response["tool_input"]
            row = {"id": case["id"], "fixture_sha256": fixture_sha,
                   "kind": case["kind"], "x": case["x"],
                   "gold": case["gold"], "prompt": case["prompt"],
                   "tool_call_emitted": response["tool_call_emitted"],
                   "tool_called": response["tool_called"],
                   "tool_input": requested,
                   "tool_input_exact": requested == case["x"],
                   "cpu_result": cpu["result"] if cpu else None,
                   "cpu_error": response.get("cpu_error"),
                   "cpu_result_exact": bool(cpu and str(cpu["result"]) == case["gold"]),
                   "cpu_for_requested_input_exact": bool(
                       cpu and cpu["result"] == (requested + 7) % 256),
                   "cpu_generated_ids": cpu["generated_ids"] if cpu else [],
                   "cpu_steps": cpu["steps"] if cpu else 0,
                   "final_exact": final["exact"],
                   "final_boxed_exact": bool(final["exact"] and
                                             final["src_exact"] == "boxed"),
                   "final_pred": final["pred_exact"],
                   "final_src": final["src_exact"],
                   "first_text": response["first_text"],
                   "final_text": response["final_text"],
                   "seconds": response["seconds"], "runtime_error": None}
        except Exception as error:
            row = {"id": case["id"], "fixture_sha256": fixture_sha,
                   "kind": case["kind"], "x": case["x"],
                   "gold": case["gold"], "prompt": case["prompt"],
                   "runtime_error": f"{type(error).__name__}: {error}"[:500]}
        with args.out.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({key: row.get(key) for key in
                          ("id", "tool_call_emitted", "tool_called", "tool_input_exact",
                           "cpu_result_exact", "cpu_error", "final_boxed_exact",
                           "runtime_error")}), flush=True)
    rows = read_jsonl(args.out)
    report = {"fixture_sha256": fixture_sha,
              "model": str(args.text_model), "device": args.device,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cpu_program_hex": PROGRAM.hex(), "cpu_graph": circuit.stats(),
              "max_text_tokens": args.max_text_tokens, **summary(rows),
              "by_kind": {kind: summary([row for row in rows if row.get("kind") == kind])
                          for kind in ("explicit", "implicit")}}
    print(json.dumps(report), flush=True)
    args.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
