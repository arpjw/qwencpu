"""Evaluate one dynamic 9B checkpoint on eight frozen byte_add7 requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_FIXTURE = (Path(__file__).resolve().parents[1] / "eval" / "frozen"
                   / "byte_bridge.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def expected_trace(x: int) -> list[int]:
    """Independent oracle used only for scoring, after model generation."""
    from gdn_cpu.byte_bridge import PROGRAM
    from gdn_cpu.qwen8 import state_token
    from gdn_cpu.qwen8_dynamic import TICK
    from gdn_cpu.reference import CPUState, step

    state = CPUState(acc=x)
    out = []
    for _ in range(2):
        state = step(state, PROGRAM)
        out.extend((TICK, state_token(state)))
    return out


def score(case: dict, response: dict) -> dict:
    from chipformer.q35.arith_bench import exact_fields

    requested = response["tool_input"]
    cpu = response["cpu"]
    if cpu is None:
        answer_text = response["first_text"] if requested is None else ""
    else:
        # Score the final assistant continuation, excluding the tool response.
        answer_text = response["final_text"].rsplit("</tool_response>", 1)[-1]
    final = exact_fields(case, answer_text)
    return {"id": case["id"], "kind": case["kind"], "x": case["x"],
            "gold": case["gold"], "prompt": case["prompt"],
            "tool_call_emitted": response["tool_call_emitted"],
            "tool_called": response["tool_called"],
            "tool_input": requested,
            "tool_input_exact": requested == case["x"],
            "cpu_result": cpu["result"] if cpu else None,
            "cpu_error": response.get("cpu_error"),
            "cpu_trace_exact": bool(cpu and cpu["raw_generated_ids"] ==
                                    expected_trace(requested)),
            "cpu_result_for_requested_exact": bool(
                cpu and cpu["result"] == ((requested + 7) & 255)),
            "cpu_result_for_question_exact": bool(
                cpu and str(cpu["result"]) == case["gold"]),
            "cpu_generated_ids": cpu["generated_ids"] if cpu else [],
            "cpu_raw_generated_ids": cpu["raw_generated_ids"] if cpu else [],
            "cpu_prompt_ids": cpu["prompt_ids"] if cpu else [],
            "cpu_states": cpu["states"] if cpu else [],
            "final_exact": final["exact"],
            "final_boxed_exact": bool(final["exact"] and
                                      final["src_exact"] == "boxed"),
            "final_pred": final["pred_exact"],
            "final_src": final["src_exact"],
            "first_text": response["first_text"],
            "final_text": response["final_text"],
            "final_answer_text": answer_text,
            "same_checkpoint": response["same_checkpoint"],
            "separate_caches": response["separate_caches"],
            "seconds": response["seconds"], "runtime_error": None}


def summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("runtime_error") is None]
    count = lambda name: sum(bool(row.get(name)) for row in completed)
    return {"cases": len(rows), "completed": len(completed),
            "tool_calls_emitted": count("tool_call_emitted"),
            "valid_tool_calls": count("tool_called"),
            "correct_tool_inputs": count("tool_input_exact"),
            "cpu_errors": count("cpu_error"),
            "cpu_traces_exact": count("cpu_trace_exact"),
            "cpu_results_for_requested_input_exact":
                count("cpu_result_for_requested_exact"),
            "cpu_results_for_question_exact":
                count("cpu_result_for_question_exact"),
            "final_answers_exact": count("final_exact"),
            "final_boxed_answers_exact": count("final_boxed_exact")}


def main() -> None:
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    from gdn_cpu.byte9b_bridge import ask, load_metadata
    from gdn_cpu.byte_bridge import PROGRAM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="dynamic merged Qwen3.5-9B snapshot")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, required=True,
                        help="resumable JSONL records; summary is adjacent JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    cases = [json.loads(line) for line in fixture_bytes.splitlines() if line.strip()]
    if len(cases) != 8 or len({row["id"] for row in cases}) != 8:
        raise ValueError("bridge fixture must contain eight unique cases")
    if args.limit is not None:
        cases = cases[:args.limit]
    meta = load_metadata(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(args.out) if args.out.exists() else []
    if any(row.get("fixture_sha256") != fixture_sha or
           row.get("model") != str(args.model) or
           row.get("max_text_tokens") != args.max_text_tokens for row in prior):
        raise ValueError("existing output differs in fixture, model, or token budget")
    done = {row["id"] for row in prior}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device).eval()
    for case in cases:
        if case["id"] in done:
            continue
        try:
            response = ask(model, tokenizer, meta, case["prompt"], device=args.device,
                           max_text_tokens=args.max_text_tokens)
            row = score(case, response)
        except Exception as error:
            row = {"id": case["id"], "kind": case["kind"], "x": case["x"],
                   "gold": case["gold"], "prompt": case["prompt"],
                   "runtime_error": f"{type(error).__name__}: {error}"[:500]}
        row.update(fixture_sha256=fixture_sha, model=str(args.model),
                   max_text_tokens=args.max_text_tokens)
        with args.out.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({key: row.get(key) for key in
                          ("id", "tool_called", "tool_input_exact", "cpu_trace_exact",
                           "cpu_result_for_question_exact", "final_boxed_exact",
                           "runtime_error")}), flush=True)
    rows = read_jsonl(args.out)
    report = {"fixture": str(args.fixture), "fixture_sha256": fixture_sha,
              "model": str(args.model), "model_loads": 1,
              "same_checkpoint": True, "separate_caches": True,
              "device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__,
              "cpu_program_hex": PROGRAM.hex(), "rom_size": meta["rom_size"],
              "max_text_tokens": args.max_text_tokens, **summary(rows),
              "by_kind": {kind: summary([row for row in rows if row.get("kind") == kind])
                          for kind in ("explicit", "implicit")}}
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
