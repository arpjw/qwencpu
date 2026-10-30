"""A small, deterministic CPU used as the circuit's independent oracle.

Instructions are one byte except those with an immediate operand, which are
two bytes. The accumulator and every RAM cell are unsigned bytes. JNZ uses an
absolute program address. There is no implicit PC wraparound.
"""

from __future__ import annotations

from dataclasses import dataclass, field


RAM_SIZE = 16
MAX_PROGRAM_SIZE = 256


@dataclass(frozen=True, slots=True)
class CPUState:
    pc: int = 0
    acc: int = 0
    ram: bytes = field(default_factory=lambda: bytes(RAM_SIZE))
    halted: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.pc <= MAX_PROGRAM_SIZE:
            raise ValueError("pc must be between 0 and 256")
        if not 0 <= self.acc <= 0xFF:
            raise ValueError("acc must be an unsigned byte")
        if len(self.ram) != RAM_SIZE:
            raise ValueError(f"ram must contain exactly {RAM_SIZE} bytes")


def step(state: CPUState, program: bytes) -> CPUState:
    """Execute exactly one instruction, returning a new state.

    A halted CPU is inert. Invalid instructions, truncated operands, and
    out-of-range fetches raise ValueError rather than silently halting.
    """
    if state.halted:
        return state
    if len(program) > MAX_PROGRAM_SIZE:
        raise ValueError("program exceeds 256 bytes")
    if state.pc >= len(program):
        raise ValueError(f"instruction fetch outside program at {state.pc}")

    opcode = program[state.pc]
    if opcode == 0x00:  # NOP
        return CPUState(state.pc + 1, state.acc, state.ram)
    if opcode == 0xFF:  # HALT
        return CPUState(state.pc, state.acc, state.ram, halted=True)

    if state.pc + 1 >= len(program):
        raise ValueError(f"missing operand for opcode 0x{opcode:02x}")
    operand = program[state.pc + 1]
    next_pc = state.pc + 2

    if opcode == 0x10:  # LDI
        return CPUState(next_pc, operand, state.ram)
    if opcode == 0x20:  # ADD immediate
        return CPUState(next_pc, (state.acc + operand) & 0xFF, state.ram)
    if opcode in (0x30, 0x40):  # STA / LDA
        if operand >= RAM_SIZE:
            raise ValueError(f"RAM address out of range: {operand}")
        if opcode == 0x40:
            return CPUState(next_pc, state.ram[operand], state.ram)
        ram = bytearray(state.ram)
        ram[operand] = state.acc
        return CPUState(next_pc, state.acc, bytes(ram))
    if opcode == 0x50:  # JNZ absolute
        return CPUState(operand if state.acc else next_pc, state.acc, state.ram)
    raise ValueError(f"unknown opcode 0x{opcode:02x} at {state.pc}")


def run(program: bytes, *, state: CPUState | None = None, max_steps: int = 1_000) -> CPUState:
    """Run until HALT, with a caller-controlled loop bound."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    current = state if state is not None else CPUState()
    for _ in range(max_steps):
        if current.halted:
            return current
        current = step(current, program)
    if current.halted:
        return current
    raise ValueError(f"program did not halt within {max_steps} steps")
