"""Run a small program through the circuit and its independent oracle."""

from __future__ import annotations

from gdn_cpu.circuit import CompiledCPU
from gdn_cpu.reference import CPUState, step


def main() -> None:
    # Count 3, 2, 1 into RAM[5], then load the final stored byte.
    program = bytes([0x10, 3, 0x30, 5, 0x20, 255, 0x50, 2, 0x40, 5, 0xFF])
    machine = CompiledCPU(program)
    oracle = CPUState()
    for count in range(1, 101):
        circuit_state = machine.step()
        oracle = step(oracle, program)
        if circuit_state != oracle:
            raise RuntimeError(f"circuit differs from reference after step {count}")
        if oracle.halted:
            print(f"verified {count} steps; accumulator={oracle.acc}; RAM[5]={oracle.ram[5]}")
            return
    raise RuntimeError("program did not halt within 100 steps")


if __name__ == "__main__":
    main()
