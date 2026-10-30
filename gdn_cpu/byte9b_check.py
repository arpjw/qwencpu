"""Check a merged 9B byte CPU against the independent reference emulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gdn_cpu.qwen8 import state_token
from gdn_cpu.reference import CPUState, step


def _head_matrix(cache, circuit, name: str):
    """Find a first value head, accounting for 9B's two values per key."""
    from chipformer.q35.ir import Head

    node = next(n for n in circuit.nodes if isinstance(n, Head) and n.name == name)
    layer = (node.level - 1) // 2
    key_index = sum(isinstance(n, Head) and n.level == node.level
                    for n in circuit.nodes[:circuit.nodes.index(node)])
    ratio = circuit.target.value_heads // circuit.target.key_heads
    return cache.layers[layer].recurrent_states[0][0, key_index * ratio].float()


def _cache_bytes(cache, circuit, name: str, rows: int, *, written: bool) -> bytes:
    from chipformer.q35.ir import A_BIT, A_REF

    matrix = _head_matrix(cache, circuit, name)
    out = bytearray(rows)
    for address in range(rows):
        row = matrix[address]
        refs = tuple(row[:2].tolist())
        allowed = ((A_REF, A_REF),) if written else ((0.0, 0.0), (A_REF, A_REF))
        if refs not in allowed:
            raise ValueError(f"{name} ballast wrong at {address}: {refs}")
        for bit in range(8):
            value = float(row[2 + bit])
            if value not in (0.0, A_BIT):
                raise ValueError(f"{name} bit wrong at {address}, bit {bit}: {value}")
            out[address] |= int(value == A_BIT) << bit
    return bytes(out)


def _rom_cache(cache, circuit, program_length: int) -> dict[str, str]:
    out = {}
    for kind in ("op", "arg"):
        values = bytearray()
        for bank in range((program_length + 127) // 128):
            count = min(128, program_length - 128 * bank)
            values += _cache_bytes(cache, circuit, f"rom.{kind}.{bank}", count,
                                   written=True)
        out[kind] = values.hex()
    return out


def check(model_dir: Path, *, program: bytes | None = None,
          max_steps: int = 100, base: Path | None = None,
          text_after_cpu: bool = True, model=None) -> dict:
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

    meta = json.loads((model_dir / "byte9b.json").read_text(encoding="utf-8"))
    mode = meta.get("mode", "state")
    if program is None:
        if meta["program_hex"] is None:
            raise ValueError("dynamic checkpoint requires --program or --fixture")
        program = bytes.fromhex(meta["program_hex"])
    elif meta["program_hex"] is not None and program.hex() != meta["program_hex"]:
        raise ValueError("program differs from this fixed-program checkpoint")
    state = CPUState()
    expected = []
    states = []
    for _ in range(max_steps):
        state = step(state, program)
        states.append(state)
        if mode in ("recurrent", "dynamic"):
            expected.append(meta["chip_token_offset"] + meta["tick_token"])
        expected.append(meta["chip_token_offset"] + state_token(state))
        if state.halted:
            break
    else:
        raise ValueError(f"program did not halt in {max_steps} steps")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = Qwen3_5ForCausalLM.from_pretrained(model_dir,
                                                   dtype=torch.bfloat16).to(device).eval()
    offset = meta["chip_token_offset"]
    if mode == "dynamic":
        from gdn_cpu.qwen8_dynamic import program_tokens

        cpu_prompt = [offset + meta["pad_token"],
                      *(offset + token for token in program_tokens(
                          program, rom_size=meta["rom_size"])),
                      offset + state_token(CPUState())]
    else:
        cpu_prompt = [offset + state_token(CPUState())]
    prompt = torch.tensor([cpu_prompt], device=device)
    with torch.no_grad():
        result = model.generate(prompt, max_new_tokens=len(expected), do_sample=False,
                                pad_token_id=offset + meta["pad_token"] if mode == "dynamic"
                                else offset, eos_token_id=None,
                                cache_implementation="dynamic")
    got = result[0, len(cpu_prompt):].tolist()
    first_bad = next((i for i, (actual, wanted) in enumerate(zip(got, expected))
                      if actual != wanted), None)
    if first_bad is None and len(got) != len(expected):
        first_bad = min(len(got), len(expected))
    report = {"program_hex": program.hex(), "steps": len(states),
              "tokens": len(expected),
              "cpu_exact": got == expected, "expected_tokens": expected,
              "generated_tokens": got, "first_bad_step": first_bad}

    if mode in ("recurrent", "dynamic"):
        from chipformer.q35.ir import Target

        if mode == "recurrent":
            from gdn_cpu.qwen8_recurrent import build as build_recurrent
        else:
            from gdn_cpu.qwen8_dynamic import build as build_dynamic

        target = Target(model.config.linear_num_key_heads,
                        model.config.linear_num_value_heads,
                        model.config.intermediate_size)
        circuit = (build_recurrent(program, target) if mode == "recurrent"
                   else build_dynamic(target, rom_size=meta["rom_size"]))
        observed_ram = []
        ram_error = None
        observed_rom = {}
        rom_error = None
        try:
            with torch.no_grad():
                output = model(input_ids=prompt, use_cache=True)
                cache = output.past_key_values
                if mode == "dynamic":
                    try:
                        observed_rom = _rom_cache(cache, circuit, len(program))
                    except (RuntimeError, ValueError) as error:
                        rom_error = f"{type(error).__name__}: {error}"[:500]
                if len(got) == len(expected):
                    for i in range(len(states)):
                        tick = torch.tensor([[got[2 * i]]], device=device)
                        output = model(input_ids=tick, past_key_values=cache,
                                       use_cache=True)
                        cache = output.past_key_values
                        observed_ram.append(_cache_bytes(cache, circuit, "ram", 16,
                                                         written=False).hex())
                        if i + 1 < len(states):
                            next_state = torch.tensor([[got[2 * i + 1]]], device=device)
                            output = model(input_ids=next_state, past_key_values=cache,
                                           use_cache=True)
                            cache = output.past_key_values
                else:
                    ram_error = "generated trace ended before the expected final state"
        except (IndexError, ValueError, RuntimeError) as error:
            ram_error = f"{type(error).__name__}: {error}"[:500]
        expected_ram = [s.ram.hex() for s in states]
        report["ram_after_ticks"] = observed_ram
        report["expected_ram_after_ticks"] = expected_ram
        report["ram_exact"] = observed_ram == expected_ram
        report["ram_error"] = ram_error
        first_bad_ram = next((i for i, (actual, wanted) in
                              enumerate(zip(observed_ram, expected_ram))
                              if actual != wanted), None)
        if first_bad_ram is None and len(observed_ram) != len(expected_ram):
            first_bad_ram = min(len(observed_ram), len(expected_ram))
        report["first_bad_ram_step"] = first_bad_ram
        if mode == "dynamic":
            report["rom_cache"] = observed_rom
            report["rom_exact"] = (observed_rom == {"op": program.hex(),
                                                   "arg": program.hex()}
                                   and rom_error is None)
            report["rom_error"] = rom_error

    after_cpu = []
    tokenizer = (AutoTokenizer.from_pretrained(base or model_dir)
                 if text_after_cpu and report["cpu_exact"] else None)
    for sample in ("Hello, world!", "The sum of 3 and 5 is") if tokenizer else ():
        text_ids = tokenizer(sample, add_special_tokens=False).input_ids
        if not text_ids:
            continue
        with_cpu = [*text_ids[:-1], *cpu_prompt, *got, text_ids[-1]]
        with torch.no_grad():
            plain = model(input_ids=torch.tensor([text_ids], device=device)).logits[
                0, -1, :meta["chip_token_offset"]].float()
            resumed = model(input_ids=torch.tensor([with_cpu], device=device)).logits[
                0, -1, :meta["chip_token_offset"]].float()
            plain_logp = torch.log_softmax(plain, -1)
            resumed_logp = torch.log_softmax(resumed, -1)
            kl = (plain_logp.exp() * (plain_logp - resumed_logp)).sum()
        after_cpu.append({"sample": sample, "trace_tokens": len(got) + len(cpu_prompt),
                          "max_abs_logit_difference": float((plain - resumed).abs().max()),
                          "kl_plain_to_after_cpu": float(kl),
                          "same_top1": int(plain.argmax()) == int(resumed.argmax())})
    report["text_after_cpu"] = after_cpu

    if base is not None:
        tokenizer = AutoTokenizer.from_pretrained(base)
        samples = ["Hello, world!", "The sum of 3 and 5 is"]
        merged_logits = []
        with torch.no_grad():
            for sample in samples:
                ids = tokenizer(sample, return_tensors="pt").input_ids.to(device)
                merged_logits.append(model(input_ids=ids).logits[0, -1, :meta["chip_token_offset"]]
                                     .float().cpu())
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        base_model = AutoModelForImageTextToText.from_pretrained(base,
                                                                 dtype=torch.bfloat16).to(device).eval()
        diffs = []
        argmax = []
        with torch.no_grad():
            for sample, merged_row in zip(samples, merged_logits, strict=True):
                ids = tokenizer(sample, return_tensors="pt").input_ids.to(device)
                base_row = (base_model(input_ids=ids).logits[0, -1,
                            :meta["chip_token_offset"]].float().cpu())
                diffs.append(float((merged_row - base_row).abs().max()))
                argmax.append(int(merged_row.argmax()) == int(base_row.argmax()))
        report["text_max_logit_difference"] = diffs
        report["text_argmax_same"] = argmax
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--program", help="program hex for a dynamic checkpoint")
    parser.add_argument("--fixture", type=Path,
                        help="JSONL program suite; loads the checkpoint once")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--no-text-after-cpu", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.fixture and args.program:
        parser.error("choose --program or --fixture")
    if args.fixture and args.base:
        parser.error("run --base on a single --program check")
    if args.fixture:
        from transformers import Qwen3_5ForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen3_5ForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(device).eval()
        cases = [json.loads(line) for line in args.fixture.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if not cases:
            raise ValueError("fixture has no program cases")
        reports = []
        for index, case in enumerate(cases):
            result = check(args.model, program=bytes.fromhex(case["program_hex"]),
                           max_steps=case.get("max_steps", args.max_steps), model=model,
                           text_after_cpu=not args.no_text_after_cpu and index == 0)
            result["id"] = case["id"]
            reports.append(result)
            print(json.dumps({"id": case["id"], "cpu_exact": result["cpu_exact"],
                              "rom_exact": result.get("rom_exact"),
                              "ram_exact": result.get("ram_exact"),
                              "steps": result["steps"]}), flush=True)
        report = {"fixture": str(args.fixture), "n_cases": len(reports),
                  "all_exact": all(row["cpu_exact"] and row.get("ram_exact", True)
                                   and row.get("rom_exact", True)
                                   for row in reports), "cases": reports}
    else:
        report = check(args.model, program=bytes.fromhex(args.program)
                       if args.program else None, max_steps=args.max_steps, base=args.base,
                       text_after_cpu=not args.no_text_after_cpu)
    if args.fixture:
        print(json.dumps({"fixture": str(args.fixture), "n_cases": report["n_cases"],
                          "all_exact": report["all_exact"]}), flush=True)
    else:
        print(json.dumps(report), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report.get("all_exact") is False or report.get("cpu_exact") is False \
            or report.get("ram_exact") is False or report.get("rom_exact") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
