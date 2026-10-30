"""Check the recurrent-register variant against the independent emulator."""

import importlib.util
import unittest

from gdn_cpu.reference import CPUState, step


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class RecurrentQwen8IRTests(unittest.TestCase):
    def test_register_and_ram_heads(self) -> None:
        from chipformer.q35.ir import A_BIT, evaluate

        from gdn_cpu.qwen8 import KAPPA, state_token
        from gdn_cpu.qwen8_recurrent import TICK, build

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
                core = build(program)
                state = CPUState()
                carry = {}
                ram_head = next(i for i, node in enumerate(core.nodes) if node.name == "ram")
                for _ in range(20):
                    next_state = step(state, program)
                    residual = evaluate(core, [state_token(state), TICK], check=True,
                                        carry=carry, kernel=True)
                    predicted = []
                    for row in residual:
                        value = 0
                        for i, expr in enumerate(core.code):
                            bit_value = row[next(iter(expr.terms))]
                            self.assertEqual(abs(bit_value), KAPPA)
                            value |= int(bit_value > 0) << i
                        predicted.append(value)
                    self.assertEqual(predicted, [TICK, state_token(next_state)])
                    rows = carry[ram_head, "S", 0]
                    observed_ram = bytes(
                        sum(int(rows[address][2 + bit] / A_BIT) << bit for bit in range(8))
                        if address in rows else 0
                        for address in range(16)
                    )
                    self.assertEqual(observed_ram, next_state.ram)
                    state = next_state
                    if state.halted:
                        break


if __name__ == "__main__":
    unittest.main()
