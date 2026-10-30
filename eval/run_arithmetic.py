"""Run the same frozen arithmetic cases in three Qwen3.5-9B conditions.

Requires the pinned qwenputer checkout and its HF dependencies on a CUDA host.
One condition runs per process to keep the model and KV cache within GPU memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


SYSTEM_TOOL = ("You have a calc tool for exact arithmetic. Use it for every "
               "arithmetic step instead of calculating by hand.")
SYSTEM_BASE = "Do every arithmetic step exactly."
SUFFIX = "\n\nGive the final answer as a single number in \\boxed{} at the end."
FROZEN_ARITHMETIC_SHA256 = "a4277e1e79229b258319454b237f2f84749c67da3ad1d60f0dcbd3fc6f4e0896"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _hash_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_resume(rows: list[dict], cases: list[dict], *, arm: str, model: str,
                    max_text_tokens: int, max_chip_tokens: int,
                    cases_sha256: str, firmware_sha256: str | None) -> set[str]:
    """Reject an output file from another run before skipping any case IDs.

    Older no-tool rows may lack file hashes. Accept them only for the original
    frozen fixture, after checking their recorded case content and settings.
    """
    by_id = {case["id"]: case for case in cases}
    done: set[str] = set()
    for row in rows:
        case_id = row.get("id")
        if case_id not in by_id or case_id in done:
            raise ValueError(f"duplicate or unexpected resume case: {case_id}")
        done.add(case_id)
        case = by_id[case_id]
        expected = {"arm": arm, "model": model,
                    "max_text_tokens": max_text_tokens,
                    "max_chip_tokens": max_chip_tokens if arm == "merged" else 0,
                    "expression": case["expression"], "gold": case["gold"],
                    "kind": case["kind"]}
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"resume case {case_id}: {key} changed")
        if "cases_sha256" in row:
            if row["cases_sha256"] != cases_sha256:
                raise ValueError(f"resume case {case_id}: fixture changed")
        elif arm != "base" or cases_sha256 != FROZEN_ARITHMETIC_SHA256:
            raise ValueError(f"resume case {case_id}: fixture hash missing")
        if firmware_sha256 is not None and row.get("firmware_sha256") != firmware_sha256:
            raise ValueError(f"resume case {case_id}: firmware changed or hash missing")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("base", "base_host", "merged"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--qwenputer-src", type=Path, required=True,
                    help="path to qwenputer/experimental/riscv")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--firmware", type=Path, required=True)
    ap.add_argument("--max-text-tokens", type=int, default=4096)
    ap.add_argument("--max-chip-tokens", type=int, default=60000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--replay", action="store_true",
                    help="also replay each merged stream through the IR and reference")
    args = ap.parse_args()
    if args.max_text_tokens < 1 or args.max_chip_tokens < 1:
        ap.error("token limits must be positive")
    cases = read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    ids = [row["id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    cases_sha256 = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    firmware_sha256 = (hashlib.sha256(args.firmware.read_bytes()).hexdigest()
                       if args.arm != "base" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = validate_resume(
        read_jsonl(args.output) if args.output.exists() else [], cases,
        arm=args.arm, model=str(args.model), max_text_tokens=args.max_text_tokens,
        max_chip_tokens=args.max_chip_tokens, cases_sha256=cases_sha256,
        firmware_sha256=firmware_sha256)
    sys.path.insert(0, str(args.qwenputer_src.resolve()))

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35 import arith_bench as AB
    from chipformer.rv import bridge as BR
    from chipformer.rv import elf, ref
    from chipformer.rv.chat import CALC_TOOL, RvChat, replay
    from chipformer.rv.evalarith import BridgeHost, run_base

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for the 9B evaluation")

    if args.arm == "merged":
        chat = RvChat(args.model, fast=True)
        tok = chat.tok
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
        model = model.to("cuda").eval()
        text = BR.Text(args.model / "tokenizer.json") if args.arm == "base_host" else None

    for case in cases:
        if case["id"] in done:
            continue
        tool = args.arm != "base"
        messages = [
            {"role": "system", "content": SYSTEM_TOOL if tool else SYSTEM_BASE},
            {"role": "user", "content": f"Compute {case['expression']}." + SUFFIX},
        ]
        if args.arm == "merged":
            prompt = chat.prompt_ids(messages, tools=(CALC_TOOL,))
        else:
            p = tok.apply_chat_template(messages, tools=[CALC_TOOL] if tool else None,
                                        add_generation_prompt=True, enable_thinking=False,
                                        tokenize=True)
            prompt = p if isinstance(p, list) else p["input_ids"]
            prompt = [int(x) for x in prompt]
        started = time.monotonic()
        extras: dict = {"calls": 0, "completed_calls": 0,
                        "text_tokens": 0, "chip_tokens": 0,
                        "truncated": False, "head_mismatch": None,
                        "lock_error": None, "replay": None, "replay_error": None}
        runtime_error = None
        try:
            if args.arm == "merged":
                machine = ref.Machine.from_program(elf.program(args.firmware))
                BR.park(machine)
                check_machine = ref.Machine.from_program(elf.program(args.firmware))
                BR.park(check_machine)
                lock = BR.Session(chat.T, check_machine)
                lock.history(prompt)
                result = chat.generate(prompt, machine, max_new=args.max_text_tokens,
                                       max_chip=args.max_chip_tokens, check_head=True, lock=lock)
                generated = result["tokens"]
                output = chat.text_of(generated)
                extras = {"calls": sum(token == chat.T.TCC for token in generated),
                          "completed_calls": len(lock.runs),
                          "text_tokens": result["text_tokens"],
                          "chip_tokens": result["chip_tokens"],
                          "truncated": (result["text_tokens"] >= args.max_text_tokens or
                                        result["chip_tokens"] >= args.max_chip_tokens),
                          "head_mismatch": result["head_mismatch"],
                          "lock_error": result["lock_error"], "replay": None,
                          "replay_error": None}
            else:
                host = BridgeHost(text, args.firmware, prompt) if text is not None else None
                generated, calls, own = run_base(model, tok, prompt, args.arm == "base_host",
                                                 args.max_text_tokens, host=host)
                output = tok.decode(generated, skip_special_tokens=False)
                extras = {"calls": calls,
                          "completed_calls": len(host.ses.runs) if host is not None else 0,
                          "text_tokens": own, "chip_tokens": 0,
                          "truncated": own >= args.max_text_tokens,
                          "head_mismatch": None, "lock_error": None, "replay": None,
                          "replay_error": None}
            score = AB.exact_fields({"gold": case["gold"]}, output)
        except Exception as error:
            runtime_error = f"{type(error).__name__}: {error}"[:500]
            generated, output = [], ""
            score = {"exact": False, "pred_exact": None, "src_exact": "runtime_error"}
        if args.arm == "merged" and args.replay and runtime_error is None:
            try:
                extras["replay"] = replay(chat, prompt, generated, args.firmware)
            except Exception as error:
                extras["replay_error"] = f"{type(error).__name__}: {error}"[:500]
        record = {"id": case["id"], "kind": case["kind"],
                  "expression": case["expression"], "gold": case["gold"],
                  "arm": args.arm, "model": str(args.model),
                  "cases_sha256": cases_sha256, "firmware_sha256": firmware_sha256,
                  "prompt_sha256": _hash_ids(prompt), "max_text_tokens": args.max_text_tokens,
                  "max_chip_tokens": args.max_chip_tokens if args.arm == "merged" else 0,
                  "seconds": round(time.monotonic() - started, 3),
                  "output": output, "generated_ids": generated,
                  "runtime_error": runtime_error,
                  **score, **extras}
        with args.output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({k: record[k] for k in
                          ("id", "arm", "exact", "calls", "lock_error", "head_mismatch",
                           "seconds")}), flush=True)


if __name__ == "__main__":
    main()
