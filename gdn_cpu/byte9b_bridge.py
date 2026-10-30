"""Route a text-selected byte_add7 call through the same dynamic 9B checkpoint.

Text selection, CPU execution, and text continuation use fresh decoder caches.
Only the CPU's generated final state supplies the tool response value.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gdn_cpu.byte_bridge import (PROGRAM, SYSTEM, _generate_text,
                                 _token_ids, parse_tool_input)


def load_metadata(model_dir: Path) -> dict:
    meta = json.loads((model_dir / "byte9b.json").read_text(encoding="utf-8"))
    if meta.get("mode") != "dynamic" or meta.get("rom_size", 0) < len(PROGRAM):
        raise ValueError("bridge requires a dynamic byte9b checkpoint with ROM size >= 3")
    return meta


def run_cpu(model, meta: dict, x: int, *, device: str) -> dict:
    """Generate ADD 7; HALT on a fresh cache of the supplied merged model."""
    import torch

    from gdn_cpu.qwen8 import STATE_VOCAB, state_token, token_state
    from gdn_cpu.qwen8_dynamic import PAD, TICK, program_tokens
    from gdn_cpu.reference import CPUState

    if not 0 <= x <= 255:
        raise ValueError("x must be an unsigned byte")
    offset = meta["chip_token_offset"]
    prompt_ids = [offset + PAD,
                  *(offset + token for token in program_tokens(
                      PROGRAM, rom_size=meta["rom_size"])),
                  offset + state_token(CPUState(acc=x))]
    prompt = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=4, do_sample=False,
                                   pad_token_id=offset + PAD, eos_token_id=None,
                                   cache_implementation="dynamic")[0, len(prompt_ids):].tolist()
    if len(generated) != 4:
        raise RuntimeError(f"CPU produced {len(generated)} tokens, expected 4")
    raw = [token - offset for token in generated]
    if raw[0] != TICK or raw[2] != TICK:
        raise RuntimeError(f"CPU missed a TICK token: {raw}")
    if not all(0 <= raw[index] < STATE_VOCAB for index in (1, 3)):
        raise RuntimeError(f"CPU missed a state token: {raw}")
    states = [token_state(raw[index]) for index in (1, 3)]
    if not states[-1].halted:
        raise RuntimeError("CPU failed to halt in two steps")
    return {"input": x, "result": states[-1].acc, "steps": 2,
            "program_hex": PROGRAM.hex(), "prompt_ids": prompt_ids,
            "generated_ids": generated, "raw_generated_ids": raw,
            "states": [{"pc": state.pc, "acc": state.acc,
                        "halted": state.halted} for state in states]}


def ask(model, tokenizer, meta: dict, question: str, *, device: str,
        max_text_tokens: int = 1024) -> dict:
    """Select a tool, execute it, and answer with one loaded 9B model."""
    if max_text_tokens < 1:
        raise ValueError("max_text_tokens must be positive")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    prompt = _token_ids(tokenizer, messages)
    start = time.monotonic()
    first_ids = _generate_text(model, prompt, tokenizer, max_text_tokens,
                               device=device, stop_on_tool=True)
    first_text = tokenizer.decode(first_ids, skip_special_tokens=False)
    x = parse_tool_input(first_text)
    emitted = "<tool_call>" in first_text and "</tool_call>" in first_text
    result = {"question": question, "tool_call_emitted": emitted,
              "tool_called": x is not None, "tool_input": x,
              "first_text": first_text, "final_text": first_text,
              "text_generated_ids": first_ids, "cpu": None,
              "same_checkpoint": True, "separate_caches": True}
    if x is None:
        result["seconds"] = round(time.monotonic() - start, 3)
        return result

    try:
        cpu = run_cpu(model, meta, x, device=device)
    except Exception as error:
        result["cpu_error"] = f"{type(error).__name__}: {error}"[:500]
        result["seconds"] = round(time.monotonic() - start, 3)
        return result
    response = f"\n<tool_response>\n{cpu['result']}\n</tool_response>\n<think>\n\n</think>\n\n"
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    # A full text-only prompt creates a fresh cache. No raw CPU ID is present.
    continuation = _generate_text(model, [*prompt, *first_ids, *response_ids],
                                  tokenizer, max_text_tokens, device=device,
                                  stop_on_tool=False)
    result.update(cpu=cpu, tool_response=response,
                  final_text=tokenizer.decode([*first_ids, *response_ids, *continuation],
                                              skip_special_tokens=False),
                  continuation_ids=continuation,
                  seconds=round(time.monotonic() - start, 3))
    return result


def main() -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="dynamic merged Qwen3.5-9B snapshot")
    parser.add_argument("--ask", default="Use byte_add7: what is 201 plus 7 modulo 256?")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    meta = load_metadata(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device).eval()
    report = ask(model, tokenizer, meta, args.ask, device=args.device,
                 max_text_tokens=args.max_text_tokens)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
