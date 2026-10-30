"""Behavioral specification for the independent CPU oracle."""

import unittest

from gdn_cpu.reference import CPUState, run, step


class ReferenceCPUTests(unittest.TestCase):
    def test_arithmetic_wrap_and_memory_round_trip(self) -> None:
        # LDI 250; ADD 10; STA 3; LDI 0; LDA 3; HALT.
        program = bytes([0x10, 250, 0x20, 10, 0x30, 3, 0x10, 0, 0x40, 3, 0xFF])
        initial = CPUState()
        final = run(program, state=initial)
        self.assertEqual((final.pc, final.acc, final.ram[3], final.halted), (10, 4, 4, True))
        self.assertEqual(initial, CPUState())  # step is pure

    def test_jnz_takes_branch_only_for_nonzero_accumulator(self) -> None:
        taken = bytes([0x10, 1, 0x50, 6, 0x10, 99, 0x20, 2, 0xFF])
        not_taken = bytes([0x10, 0, 0x50, 6, 0x10, 99, 0xFF])
        self.assertEqual(run(taken).acc, 3)
        self.assertEqual(run(not_taken).acc, 99)

    def test_halt_is_inert(self) -> None:
        program = bytes([0xFF])
        halted = step(CPUState(), program)
        self.assertEqual(step(halted, program), halted)

    def test_invalid_fetch_operand_opcode_and_address_fail(self) -> None:
        cases = (
            (CPUState(), b"", "fetch"),
            (CPUState(), bytes([0x10]), "missing operand"),
            (CPUState(), bytes([0x7F, 0]), "unknown opcode"),
            (CPUState(), bytes([0x30, 16]), "RAM address"),
        )
        for state, program, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                step(state, program)

    def test_loop_bound_catches_program_without_halt(self) -> None:
        program = bytes([0x10, 1, 0x50, 2])
        with self.assertRaisesRegex(ValueError, "did not halt"):
            run(program, max_steps=16)


if __name__ == "__main__":
    unittest.main()
