"""Score a model-side 8-bit CPU through a persistent JSON-lines adapter.

For each input line the adapter receives {id, program_hex, max_steps}; it must
return {id, trace}, where trace is the initial state followed by every executed
state through HALT. No Python reference state is supplied to the adapter.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from eval.run_arithmetic import read_jsonl


def _normalise(trace: list[dict]) -> list[dict]:
    return [{"pc": int(s["pc"]), "acc": int(s["acc"]),
             "ram_hex": str(s["ram_hex"]).lower(), "halted": bool(s["halted"])}
            for s in trace]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--adapter-cmd", required=True,
                    help="persistent process reading one JSON case and writing one JSON reply")
    args = ap.parse_args()
    cases = read_jsonl(args.cases)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in read_jsonl(args.output)} if args.output.exists() else set()
    with subprocess.Popen(shlex.split(args.adapter_cmd), stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        try:
            for case in cases:
                if case["id"] in done:
                    continue
                request = {key: case[key] for key in ("id", "program_hex", "max_steps")}
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"adapter exited before replying to {case['id']}")
                reply = json.loads(line)
                if reply.get("id") != case["id"]:
                    raise ValueError(f"adapter replied with wrong ID for {case['id']}")
                trace = _normalise(reply["trace"])
                expected = case["expected_trace"]
                first_bad = next((i for i, (got, want) in
                                  enumerate(zip(trace, expected, strict=False))
                                  if got != want), None)
                if first_bad is None and len(trace) != len(expected):
                    first_bad = min(len(trace), len(expected))
                result = {"id": case["id"], "exact_trace": first_bad is None,
                          "first_bad_step": first_bad, "steps": len(trace) - 1,
                          "expected_steps": len(expected) - 1,
                          "trace": trace}
                with args.output.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(result) + "\n")
                print(json.dumps({key: result[key] for key in
                                  ("id", "exact_trace", "first_bad_step")}), flush=True)
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
