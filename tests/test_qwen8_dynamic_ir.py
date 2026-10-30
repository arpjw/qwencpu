"""A shared programmable-ROM graph must execute different loaded programs."""

import importlib.util
import unittest

from gdn_cpu.reference import CPUState, step


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class DynamicROMIRTests(unittest.TestCase):
    def test_three_programs_one_graph_with_high_bank_addresses(self) -> None:
        from chipformer.q35.ir import evaluate

        from gdn_cpu.qwen8 import KAPPA, state_token
        from gdn_cpu.qwen8_dynamic import TICK, build, program_tokens

        crossed = bytearray((0x10, 1, 0x50, 127))
        crossed.extend(bytes(127 - len(crossed)))
        crossed.extend((0x10, 7, 0x20, 3, 0xFF))
        near_end = bytearray((0x10, 1, 0x50, 254))
        near_end.extend(bytes(254 - len(near_end)))
        near_end.extend((0x00, 0xFF))
        programs = (bytes.fromhex("1001200230004000ff"), bytes(crossed), bytes(near_end))
        circuit = build()
        self.assertEqual(circuit.stats()["heads"], 6)

        for program in programs:
            with self.subTest(program_length=len(program)):
                state = CPUState()
                tokens = [*program_tokens(program), state_token(state)]
                tick_positions = []
                for _ in range(20):
                    state = step(state, program)
                    tick_positions.append(len(tokens))
                    tokens.extend((TICK, state_token(state)))
                    if state.halted:
                        break
                self.assertTrue(state.halted)
                residual = evaluate(circuit, tokens, check=True, kernel=True)
                predicted = []
                for position in tick_positions:
                    bits = []
                    for expr in circuit.code:
                        value = residual[position, next(iter(expr.terms))]
                        self.assertEqual(abs(value), KAPPA)
                        bits.append(int(value > 0))
                    predicted.append(sum(bit << i for i, bit in enumerate(bits)))
                self.assertEqual(predicted, [tokens[position + 1]
                                             for position in tick_positions])


if __name__ == "__main__":
    unittest.main()
