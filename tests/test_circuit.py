"""Behavioral checks against the separately written reference emulator."""

from __future__ import annotations

import random
import unittest

from gdn_cpu.circuit import CompiledCPU, DeltaNetMemory, add_bytes, bits_of, byte_of
from gdn_cpu.reference import CPUState, step


class MemoryTests(unittest.TestCase):
    def test_addressed_overwrite_survives_long_history(self) -> None:
        memory = DeltaNetMemory.zeroed(16)
        expected = [0] * 16
        rng = random.Random(1925)
        for _ in range(1_000):
            address = rng.randrange(16)
            value = rng.randrange(256)
            key = tuple(int(i == address) for i in range(16))
            memory.write(key, bits_of(value))
            expected[address] = value
            self.assertEqual(byte_of(memory.read(key)), value)
        self.assertEqual([byte_of(tuple(row)) for row in memory.rows], expected)

    def test_masked_write_does_not_change_memory(self) -> None:
        memory = DeltaNetMemory.zeroed(2)
        key = (1, 0)
        memory.write(key, bits_of(23))
        memory.write(key, bits_of(99), beta=0)
        self.assertEqual(byte_of(memory.read(key)), 23)


class CircuitTests(unittest.TestCase):
    def test_adder_boundary_cases(self) -> None:
        for a, b in ((0, 0), (255, 1), (128, 128), (77, 179), (255, 255)):
            with self.subTest(a=a, b=b):
                self.assertEqual(byte_of(add_bytes(bits_of(a), bits_of(b))), (a + b) & 255)

    def test_branch_ram_and_wrap_match_reference_at_each_step(self) -> None:
        # Counts down from 3, storing each intermediate value in RAM[5].
        program = bytes([0x10, 3, 0x30, 5, 0x20, 255, 0x50, 2, 0x40, 5, 0xFF])
        circuit = CompiledCPU(program)
        oracle = CPUState()
        for _ in range(12):
            oracle = step(oracle, program)
            self.assertEqual(circuit.step(), oracle)
            if oracle.halted:
                break
        self.assertTrue(oracle.halted)
        self.assertEqual(oracle.ram[5], 1)
        self.assertEqual(circuit.step(), oracle)

    def test_random_straight_line_programs_match_oracle(self) -> None:
        rng = random.Random(351)
        for case in range(4):
            program = bytearray()
            for _ in range(3):
                op = rng.choice((0x10, 0x20, 0x30, 0x40))
                operand = rng.randrange(16 if op in (0x30, 0x40) else 256)
                program.extend((op, operand))
            program.append(0xFF)
            machine = CompiledCPU(bytes(program))
            oracle = CPUState()
            for _ in range(4):
                oracle = step(oracle, program)
                self.assertEqual(machine.step(), oracle, f"case {case}")

    def test_invalid_memory_address_matches_oracle_error(self) -> None:
        program = bytes([0x30, 0x10, 0xFF])
        with self.assertRaisesRegex(ValueError, "RAM address out of range"):
            step(CPUState(), program)
        with self.assertRaisesRegex(ValueError, "RAM address out of range"):
            CompiledCPU(program).step()

    def test_program_counter_can_reach_256_without_wrapping(self) -> None:
        program = bytes([0x00] * 254 + [0x10, 0x7F])
        machine = CompiledCPU(program)
        oracle = CPUState(pc=254)
        machine.memory.write(machine._register_key(machine.PC), bits_of(254))
        oracle = step(oracle, program)
        self.assertEqual(machine.step(), oracle)
        self.assertEqual(oracle.pc, 256)
        with self.assertRaisesRegex(ValueError, "instruction fetch outside program"):
            machine.step()


if __name__ == "__main__":
    unittest.main()
