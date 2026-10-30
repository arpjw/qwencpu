"""Summarise arithmetic and program JSONL results without hiding missing cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.run_arithmetic import read_jsonl


def arithmetic_summary(cases: list[dict], paths: list[Path]) -> dict:
    expected = {case["id"]: case for case in cases}
    if len(expected) != len(cases):
        raise ValueError("duplicate case IDs in arithmetic fixture")
    summary: dict = {}
    tool_prompts: dict[str, dict[str, str]] = {}
    for path in paths:
        rows = read_jsonl(path)
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) - expected.keys():
            raise ValueError(f"duplicate or unexpected IDs in {path}")
        for row in rows:
            case = expected[row["id"]]
            if any(row.get(field) != case[field] for field in ("kind", "expression", "gold")):
                raise ValueError(f"case content changed in {path}: {row['id']}")
        arms = {row["arm"] for row in rows}
        if len(arms) != 1:
            raise ValueError(f"expected one arm in {path}, got {arms}")
        arm = arms.pop() if arms else path.stem
        if arm in summary:
            raise ValueError(f"duplicate arithmetic arm: {arm}")
        if arm in ("base_host", "merged"):
            tool_prompts[arm] = {row["id"]: row["prompt_sha256"] for row in rows}
        n_exact = sum(bool(row["exact"]) for row in rows)
        n_boxed_exact = sum(bool(row["exact"]) and row.get("src_exact") == "boxed"
                            for row in rows)
        n_called_tool = sum(row.get("calls", 0) > 0 for row in rows)
        summary[arm] = {"n_expected": len(expected), "n_run": len(rows),
                        "n_missing": len(expected.keys() - set(ids)),
                        "n_exact": n_exact,
                        "n_boxed_exact": n_boxed_exact,
                        "exact_rate_run": n_exact / len(rows) if rows else None,
                        "boxed_exact_rate_run": n_boxed_exact / len(rows) if rows else None,
                        "n_called_tool": n_called_tool,
                        "n_completed_tool": sum(row.get("completed_calls", 0) > 0
                                                for row in rows),
                        "tool_invocation_rate_run": n_called_tool / len(rows) if rows else None,
                        "n_parse_failures": sum(row.get("pred_exact") is None and
                                                row.get("runtime_error") is None for row in rows),
                        "n_runtime_errors": sum(row.get("runtime_error") is not None
                                                for row in rows),
                        "n_hit_limit": sum(
                            row.get("text_tokens", 0) >= row.get("max_text_tokens", 10**9)
                            or (row.get("max_chip_tokens", 0) > 0 and
                                row.get("chip_tokens", 0) >= row["max_chip_tokens"])
                            for row in rows),
                        "n_lock_errors": sum(row.get("lock_error") is not None for row in rows),
                        "n_replay_errors": sum(row.get("replay_error") is not None for row in rows),
                        "n_replay_failures": sum(row.get("replay") is not None and
                                                 row["replay"].get("ok") is False
                                                 for row in rows),
                        "n_head_mismatches": sum(row.get("head_mismatch") or 0 for row in rows),
                        "text_tokens": sum(row.get("text_tokens", 0) for row in rows),
                        "chip_tokens": sum(row.get("chip_tokens", 0) for row in rows),
                        "seconds": round(sum(row.get("seconds", 0) for row in rows), 3)}
    if "base_host" in tool_prompts and "merged" in tool_prompts:
        host, merged = tool_prompts["base_host"], tool_prompts["merged"]
        mismatches = sorted(case_id for case_id in host.keys() & merged.keys()
                            if host[case_id] != merged[case_id])
        if mismatches:
            raise ValueError(f"tool-arm prompts differ for: {', '.join(mismatches)}")
    return summary


def program_summary(cases: list[dict], path: Path) -> dict:
    rows = read_jsonl(path)
    expected = {case["id"] for case in cases}
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) - expected:
        raise ValueError(f"duplicate or unexpected IDs in {path}")
    return {"n_expected": len(expected), "n_run": len(rows),
            "n_missing": len(expected - set(ids)),
            "n_exact_trace": sum(bool(row["exact_trace"]) for row in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arithmetic-cases", type=Path, required=True)
    ap.add_argument("--arithmetic-results", type=Path, nargs="*", default=[])
    ap.add_argument("--program-cases", type=Path)
    ap.add_argument("--program-results", type=Path)
    args = ap.parse_args()
    report = {"arithmetic": arithmetic_summary(read_jsonl(args.arithmetic_cases),
                                                args.arithmetic_results)}
    if args.program_cases and args.program_results:
        report["programs"] = program_summary(read_jsonl(args.program_cases),
                                             args.program_results)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
