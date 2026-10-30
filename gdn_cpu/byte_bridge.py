"""A host-routed toy tool backed by the standalone Qwen3.5 byte CPU.

The text model decides whether to call ``byte_add7``. If it does, the host
parses an unsigned byte, turns it into an initial CPU state token, and runs a
separate small Qwen3.5 model whose fixed program is ADD 7; HALT. The host
decodes the final model-generated accumulator and feeds it back as a tool
response. The host does not calculate the arithmetic result.

This is a two-model bridge. It does not put the tool-selection policy or a
runtime program loader into the 9B byte-CPU checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


PROGRAM = bytes.fromhex("2007ff")
TOOL = {"type": "function", "function": {
    "name": "byte_add7",
    "description": ("Add 7 to an unsigned 8-bit integer, wrapping modulo 256. "
                    "Use this for exact byte arithmetic."),
    "parameters": {"type": "object", "properties": {
        "x": {"type": "integer", "minimum": 0, "maximum": 255,
              "description": "The unsigned byte to add 7 to"}}, "required": ["x"]}}}
SYSTEM = ("For questions asking for an unsigned byte plus 7 modulo 256, "
          "call byte_add7 with the input byte.")


def parse_tool_input(generated_text: str) -> int | None:
    """Return the model's requested byte, or None when it made no valid call."""
    block = generated_text.rsplit("<tool_call>", 1)[-1]
    if block == generated_text or "<function=byte_add7>" not in block:
        return None
    if "</tool_call>" not in block:
        return None
    body = block.split("</tool_call>", 1)[0]
    match = re.search(r"<parameter=x>\s*([0-9]{1,3})\s*</parameter>", body)
    if match is None:
        json_match = re.search(r"\{[^{}]*\}", body)
        if json_match is None:
            return None
        try:
            value = json.loads(json_match.group()).get("x")
        except json.JSONDecodeError:
            return None
        if type(value) is not int:
            return None
    else:
        value = int(match.group(1))
    return value if 0 <= value <= 255 else None


def run_cpu(model, x: int, *, device: str, max_steps: int = 4) -> dict:
    """Execute the fixed program via stock greedy generation, with no oracle."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    from gdn_cpu.qwen8 import STATE_VOCAB, state_token, token_state
    from gdn_cpu.qwen8_recurrent import PAD, TICK
    from gdn_cpu.reference import CPUState

    if not 0 <= x <= 255:
        raise ValueError("x must be an unsigned byte")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    class HaltOnState(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            token = int(input_ids[0, -1])
            return token < STATE_VOCAB and bool(token & (1 << 16))

    prompt = torch.tensor([[PAD, state_token(CPUState(acc=x))]], device=device)
    with torch.no_grad():
        tokens = model.generate(prompt, max_new_tokens=2 * max_steps, do_sample=False,
                                pad_token_id=PAD, eos_token_id=None,
                                stopping_criteria=StoppingCriteriaList([HaltOnState()]),
                                cache_implementation="dynamic")[0, 2:].tolist()
    if len(tokens) < 2 or len(tokens) % 2:
        raise RuntimeError(f"CPU produced an incomplete trace: {tokens}")
    for index, token in enumerate(tokens):
        if index % 2 == 0 and token != TICK:
            raise RuntimeError(f"expected TICK at output {index}, got {token}")
        if index % 2 == 1 and not 0 <= token < STATE_VOCAB:
            raise RuntimeError(f"expected a state at output {index}, got {token}")
    state = token_state(tokens[-1])
    if not state.halted:
        raise RuntimeError("CPU did not halt within the step limit")
    return {"input": x, "result": state.acc, "steps": len(tokens) // 2,
            "generated_ids": tokens, "final_pc": state.pc}


def _token_ids(tokenizer, messages: list[dict]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tools=[TOOL], add_generation_prompt=True,
                                        enable_thinking=False, tokenize=True)
    return [int(x) for x in (ids if isinstance(ids, list) else ids["input_ids"])]


def _generate_text(model, ids: list[int], tokenizer, max_new_tokens: int,
                   *, device: str, stop_on_tool: bool) -> list[int]:
    import torch

    ends = [tokenizer.convert_tokens_to_ids(name) for name in ("<|im_end|>", "<|endoftext|>")]
    if stop_on_tool:
        ends.append(tokenizer.convert_tokens_to_ids("</tool_call>"))
    ends = sorted({int(token) for token in ends if isinstance(token, int) and token >= 0})
    with torch.no_grad():
        sequence = model.generate(torch.tensor([ids], device=device),
                                  max_new_tokens=max_new_tokens, do_sample=False,
                                  eos_token_id=ends)[0].tolist()
    return sequence[len(ids):]


def ask(text_model, cpu_model, tokenizer, question: str, *, device: str,
        max_text_tokens: int = 1024) -> dict:
    """Let the text model choose the tool, then route its call to the CPU model."""
    if max_text_tokens < 1:
        raise ValueError("max_text_tokens must be positive")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    prompt = _token_ids(tokenizer, messages)
    start = time.monotonic()
    first_ids = _generate_text(text_model, prompt, tokenizer, max_text_tokens,
                               device=device, stop_on_tool=True)
    first_text = tokenizer.decode(first_ids, skip_special_tokens=False)
    x = parse_tool_input(first_text)
    emitted = "<tool_call>" in first_text and "</tool_call>" in first_text
    result = {"question": question, "tool_call_emitted": emitted,
              "tool_called": x is not None,
              "tool_input": x, "cpu": None, "first_text": first_text,
              "final_text": first_text, "text_generated_ids": first_ids}
    if x is None:
        result["seconds"] = round(time.monotonic() - start, 3)
        return result

    try:
        cpu = run_cpu(cpu_model, x, device=device)
    except Exception as error:
        result["cpu_error"] = f"{type(error).__name__}: {error}"[:500]
        result["seconds"] = round(time.monotonic() - start, 3)
        return result
    response = f"\n<tool_response>\n{cpu['result']}\n</tool_response>\n<think>\n\n</think>\n\n"
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    continuation = _generate_text(text_model, [*prompt, *first_ids, *response_ids],
                                  tokenizer, max_text_tokens, device=device,
                                  stop_on_tool=False)
    result.update(cpu=cpu,
                  final_text=tokenizer.decode([*first_ids, *response_ids, *continuation],
                                              skip_special_tokens=False),
                  continuation_ids=continuation,
                  seconds=round(time.monotonic() - start, 3))
    return result


def main() -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from gdn_cpu.qwen8_recurrent import build_standalone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_model", type=Path, help="base Qwen3.5-9B snapshot")
    parser.add_argument("--ask", default="Use byte_add7: what is 201 plus 7 modulo 256?")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModelForImageTextToText.from_pretrained(
        args.text_model, dtype=torch.bfloat16).to(args.device).eval()
    cpu_model, _ = build_standalone(PROGRAM)
    cpu_model.to(args.device).eval()
    report = ask(text_model, cpu_model, tokenizer, args.ask, device=args.device,
                 max_text_tokens=args.max_text_tokens)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
