"""Differential traces for the Qwen3.5 circuit IR, when its backend is installed."""

import importlib.util
import random
import unittest

from gdn_cpu.reference import CPUState, step


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class Qwen8IRTests(unittest.TestCase):
    def assert_program(self, program: bytes, max_steps: int = 20) -> None:
        from chipformer.q35.ir import evaluate

        from gdn_cpu.qwen8 import KAPPA, build, state_token

        circuit = build(program)
        state = CPUState()
        before, after = [], []
        for _ in range(max_steps):
            before.append(state)
            state = step(state, program)
            after.append(state)
            if state.halted:
                before.append(state)
                after.append(step(state, program))
                break
        residual = evaluate(circuit, [state_token(s) for s in before], check=True)
        predicted = []
        for row in residual:
            value = 0
            for i, expr in enumerate(circuit.code):
                bit_value = row[next(iter(expr.terms))]
                self.assertEqual(abs(bit_value), KAPPA)
                value |= int(bit_value > 0) << i
            predicted.append(value)
        self.assertEqual(predicted, [state_token(s) for s in after])

    def test_reference_traces(self) -> None:
        programs = [
            bytes.fromhex("1001200230004000ff"),
            bytes.fromhex("100050082001ff00002001ff"),
            bytes.fromhex("10015008109900002001ff"),
            bytes.fromhex("10ff2002ff"),
            bytes.fromhex("10073002100930024002ff"),
            bytes.fromhex("10015002"),
        ]
        for program in programs:
            with self.subTest(program=program.hex()):
                self.assert_program(program)

    def test_random_straight_line_programs(self) -> None:
        for seed in range(32):
            rng = random.Random(seed)
            program = bytearray()
            for _ in range(30):
                op = rng.choice((0x00, 0x10, 0x20, 0x30, 0x40))
                program.append(op)
                if op in (0x10, 0x20):
                    program.append(rng.randrange(256))
                elif op in (0x30, 0x40):
                    program.append(rng.randrange(16))
            program.append(0xFF)
            with self.subTest(seed=seed):
                self.assert_program(bytes(program), max_steps=31)


if __name__ == "__main__":
    unittest.main()
