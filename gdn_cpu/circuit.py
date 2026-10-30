"""A fixed-weight bit circuit backed by addressable Gated DeltaNet state.

The CPU datapath uses affine/ReLU gates. Python controls evaluation of this
fixed circuit and validates input programs; it does not implement the ISA's
arithmetic or branches. This is a proof of the memory and circuit construction,
not yet a transformer checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from gdn_cpu.reference import CPUState, MAX_PROGRAM_SIZE, RAM_SIZE


Bit = int
ByteBits = tuple[Bit, ...]


def relu(value: int) -> int:
    return max(0, value)


def and_bit(left: Bit, right: Bit) -> Bit:
    return relu(left + right - 1)


def or_bit(left: Bit, right: Bit) -> Bit:
    return left + right - and_bit(left, right)


def xor_bit(left: Bit, right: Bit) -> Bit:
    return left + right - 2 * and_bit(left, right)


def invert(bit: Bit) -> Bit:
    return 1 - bit


def bits_of(value: int) -> ByteBits:
    if not 0 <= value <= 255:
        raise ValueError("expected an unsigned byte")
    return tuple((value >> shift) & 1 for shift in range(8))


def byte_of(bits: ByteBits) -> int:
    return sum(bit << shift for shift, bit in enumerate(bits))


ZERO: ByteBits = (0,) * 8
ONE: ByteBits = bits_of(1)
TWO: ByteBits = bits_of(2)


def add_bytes(left: ByteBits, right: ByteBits) -> ByteBits:
    """Eight ripple-carry full adders, each made from fixed affine/ReLU gates."""
    return add_bytes_with_carry(left, right)[0]


def add_bytes_with_carry(left: ByteBits, right: ByteBits) -> tuple[ByteBits, Bit]:
    """Return the byte sum and carry into the ninth program-counter bit."""
    result: list[Bit] = []
    carry = 0
    for a, b in zip(left, right, strict=True):
        pair = xor_bit(a, b)
        result.append(xor_bit(pair, carry))
        carry = or_bit(and_bit(a, b), and_bit(pair, carry))
    return tuple(result), carry


def eq_byte(left: ByteBits, right: ByteBits) -> Bit:
    equal = 1
    for a, b in zip(left, right, strict=True):
        equal = and_bit(equal, invert(xor_bit(a, b)))
    return equal


def nonzero(value: ByteBits) -> Bit:
    result = 0
    for bit in value:
        result = or_bit(result, bit)
    return result


def choose(mask: Bit, yes: ByteBits, no: ByteBits) -> ByteBits:
    return tuple(
        and_bit(mask, a) + and_bit(invert(mask), b)
        for a, b in zip(yes, no, strict=True)
    )


def one_hot(address: ByteBits, size: int) -> tuple[Bit, ...]:
    return tuple(eq_byte(address, bits_of(i)) for i in range(size))


@dataclass(slots=True)
class DeltaNetMemory:
    """S' = alpha*S + beta*k*(v - alpha*k^T*S), with one-hot k.

    Keeping alpha=beta=1 turns a row into exact overwriteable RAM. A beta
    of zero suppresses a write without changing the state.
    """

    rows: list[list[int]]

    @classmethod
    def zeroed(cls, row_count: int, width: int = 8) -> DeltaNetMemory:
        return cls([[0] * width for _ in range(row_count)])

    def read(self, key: tuple[Bit, ...]) -> ByteBits:
        if len(key) != len(self.rows):
            raise ValueError("key width does not match memory")
        return tuple(sum(key[row] * value for row, value in enumerate(column))
                     for column in zip(*self.rows, strict=True))

    def write(self, key: tuple[Bit, ...], value: ByteBits, *, beta: Bit = 1) -> None:
        old = self.read(key)
        for row, selected in enumerate(key):
            for col, (new_bit, old_bit) in enumerate(zip(value, old, strict=True)):
                self.rows[row][col] += beta * selected * (new_bit - old_bit)


class CompiledCPU:
    """Straight-line circuit for a tiny accumulator ISA and 16 bytes of RAM."""

    ACC = RAM_SIZE
    PC = RAM_SIZE + 1
    HALTED = RAM_SIZE + 2
    PC_HIGH = RAM_SIZE + 3

    def __init__(self, program: bytes) -> None:
        if not program or len(program) > MAX_PROGRAM_SIZE:
            raise ValueError("program length must be 1..256 bytes")
        self.program = bytes(program)
        self.rom = tuple(bits_of(byte) for byte in program) + (ZERO,) * (256 - len(program))
        self.memory = DeltaNetMemory.zeroed(RAM_SIZE + 4)

    def _register_key(self, row: int) -> tuple[Bit, ...]:
        return tuple(int(i == row) for i in range(RAM_SIZE + 4))

    def _rom_read(self, address: ByteBits) -> ByteBits:
        key = one_hot(address, 256)
        return tuple(
            sum(selected * row[col] for selected, row in zip(key, self.rom, strict=True))
            for col in range(8)
        )

    def snapshot(self) -> CPUState:
        return CPUState(
            pc=byte_of(self.memory.read(self._register_key(self.PC)))
            + 256 * self.memory.read(self._register_key(self.PC_HIGH))[0],
            acc=byte_of(self.memory.read(self._register_key(self.ACC))),
            ram=bytes(byte_of(self.memory.read(self._register_key(i))) for i in range(RAM_SIZE)),
            halted=bool(self.memory.read(self._register_key(self.HALTED))[0]),
        )

    def step(self) -> CPUState:
        """Run one compiled circuit transition and expose its state for checking."""
        before = self.snapshot()
        if before.halted:
            return before
        if before.pc >= len(self.program):
            raise ValueError(f"instruction fetch outside program at {before.pc}")

        pc = self.memory.read(self._register_key(self.PC))
        acc = self.memory.read(self._register_key(self.ACC))
        opcode = self._rom_read(pc)
        operand = self._rom_read(add_bytes(pc, ONE))
        nop, ldi, add, sta, lda, jnz, halt = (
            eq_byte(opcode, bits_of(code))
            for code in (0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0xFF)
        )
        if not (nop or ldi or add or sta or lda or jnz or halt):
            raise ValueError(f"unknown opcode 0x{byte_of(opcode):02x} at {before.pc}")
        if not (nop or halt) and before.pc + 1 >= len(self.program):
            raise ValueError(f"missing operand for opcode 0x{byte_of(opcode):02x}")
        if (sta or lda) and byte_of(operand) >= RAM_SIZE:
            raise ValueError(f"RAM address out of range: {byte_of(operand)}")

        ram_key = one_hot(operand, RAM_SIZE)
        ram_value = self.memory.read(ram_key + (0, 0, 0, 0))
        next_acc = choose(ldi, operand, acc)
        next_acc = choose(add, add_bytes(acc, operand), next_acc)
        next_acc = choose(lda, ram_value, next_acc)

        fallthrough, carry = add_bytes_with_carry(pc, choose(nop, ONE, TWO))
        branch = and_bit(jnz, nonzero(acc))
        next_pc = choose(branch, operand, fallthrough)
        next_pc = choose(halt, pc, next_pc)
        next_pc_high = and_bit(invert(or_bit(branch, halt)), carry)

        self.memory.write(ram_key + (0, 0, 0, 0), acc, beta=sta)
        self.memory.write(self._register_key(self.ACC), next_acc)
        self.memory.write(self._register_key(self.PC), next_pc)
        self.memory.write(self._register_key(self.PC_HIGH), bits_of(next_pc_high))
        self.memory.write(self._register_key(self.HALTED), bits_of(halt))
        return self.snapshot()
