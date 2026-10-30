"""Build and execute the byte CPU in stock Qwen3.5 layers.

Run with the Qwenputer experimental/riscv package installed, for example:
  PYTHONPATH=/path/to/gdn-cpu-lab uv run --extra hf \
    python -m gdn_cpu.qwen8_check --program 1001200230004000ff
"""

from __future__ import annotations

import argparse
import json

import torch

from gdn_cpu.qwen8 import STATE_VOCAB, build_standalone, state_token
from gdn_cpu.reference import CPUState, RAM_SIZE, step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", default="1001200230004000ff", help="program bytes as hex")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recurrent", action="store_true",
                        help="use the variant with PC/accumulator/halt in GDN state")
    args = parser.parse_args()
    program = bytes.fromhex(args.program)
    if args.recurrent:
        from gdn_cpu.qwen8_recurrent import PAD, TICK
        from gdn_cpu.qwen8_recurrent import build_standalone as build_recurrent

        model, circuit = build_recurrent(program)
        pad = PAD
    else:
        model, circuit = build_standalone(program)
        pad = STATE_VOCAB
    model.to(args.device)
    state = CPUState()
    expected = []
    for _ in range(args.steps):
        state = step(state, program)
        if args.recurrent:
            expected.append(TICK)
        expected.append(state_token(state))
    prompt = torch.tensor([[pad, state_token(CPUState())]], device=args.device)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=len(expected), do_sample=False,
                                   pad_token_id=pad, eos_token_id=None,
                                   cache_implementation="dynamic")[0].tolist()[2:]
    ram_exact = None
    if args.recurrent and generated == expected:
        from chipformer.q35.ir import A_BIT, A_REF, Head

        ram_node = next(n for n in circuit.nodes if n.name == "ram")
        layer = (ram_node.level - 1) // 2
        head = sum(isinstance(n, Head) and n.level == ram_node.level
                   for n in circuit.nodes[:circuit.nodes.index(ram_node)])
        cache = None
        token_input = prompt
        reference = CPUState()
        ram_exact = True
        with torch.no_grad():
            for index, generated_token in enumerate(generated):
                output = model(token_input, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                if int(output.logits[0, -1].argmax()) != generated_token:
                    ram_exact = False
                    break
                if index % 2:  # the input was TICK, which committed one instruction
                    reference = step(reference, program)
                    matrix = cache.layers[layer].recurrent_states[0][0, head].float()
                    observed = bytearray(RAM_SIZE)
                    for address in range(RAM_SIZE):
                        row = matrix[address]
                        if tuple(row[:2].tolist()) not in ((0.0, 0.0),
                                                             (A_REF, A_REF)):
                            ram_exact = False
                        for bit in range(8):
                            value = float(row[2 + bit])
                            if value not in (0.0, A_BIT):
                                ram_exact = False
                            observed[address] |= int(value == A_BIT) << bit
                    if bytes(observed) != reference.ram:
                        ram_exact = False
                token_input = torch.tensor([[generated_token]], device=args.device)
    result = {"program": program.hex(), "graph": circuit.stats(),
              "model_layers": model.config.num_hidden_layers,
              "hidden": model.config.hidden_size, "recurrent": args.recurrent,
              "expected": expected, "generated": generated,
              "exact": generated == expected, "ram_exact": ram_exact}
    print(json.dumps(result))
    if not result["exact"] or ram_exact is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
