"""Compile the lab's byte CPU into Qwen3.5 circuit IR.

This uses Qwenputer's generic Qwen3.5 graph, gate and weight emitter as a
backend. The ISA, fixed program decoder, RAM layout and state transition here
are separate from its CHIP-8 and RV32 machines. A trace token carries the
current PC, accumulator and halt bit. The next token is predicted by the
circuit; writable RAM is held by one Gated DeltaNet head across tokens.

The program is fixed when the graph is built. This first model target covers
valid traces with PC <= 255; malformed instructions remain the reference
emulator's responsibility.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from gdn_cpu.reference import CPUState, MAX_PROGRAM_SIZE, RAM_SIZE


STATE_VOCAB = 1 << 17
CODE_AMP = 128
KAPPA = 2.0**-20
OPCODES = (0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0xFF)


def state_token(state: CPUState) -> int:
    if not 0 <= state.pc < 256:
        raise ValueError("this trace encoding requires pc < 256")
    return state.pc | state.acc << 8 | int(state.halted) << 16


def token_state(token: int, ram: bytes = bytes(RAM_SIZE)) -> CPUState:
    if not 0 <= token < STATE_VOCAB:
        raise ValueError("state token out of range")
    return CPUState(token & 255, (token >> 8) & 255, ram, bool(token >> 16))


class ByteCore(Circ):
    def __init__(self, program: bytes, target: Target | None = None) -> None:
        if not program or len(program) > MAX_PROGRAM_SIZE:
            raise ValueError("program length must be 1..256 bytes")
        super().__init__(target or Target())
        self.program = bytes(program)
        f = self.token_feature
        self.pc = [f(f"pc.b{i}", lambda t, i=i: (t >> i) & 1, full=True)
                   for i in range(8)]
        self.acc = [f(f"acc.b{i}", lambda t, i=i: (t >> (8 + i)) & 1, full=True)
                    for i in range(8)]
        self.halted = f("halted", lambda t: (t >> 16) & 1, full=True)
        self.code = [f(f"code.b{i}", lambda t, i=i:
                       CODE_AMP * (2 * ((t >> i) & 1) - 1), full=True)
                     for i in range(17)]
        self.keep = {next(iter(e.terms)) for e in self.code}


def _add8(c: ByteCore, a: list[Expr], b: list[Expr]) -> list[Expr]:
    carry: Expr = Expr()
    result = []
    for i, (left, right) in enumerate(zip(a, b, strict=True)):
        ab = c.and_([left, right], f"add.ab{i}")
        pair = left + right - 2 * ab
        pc = c.and_([pair, carry], f"add.pc{i}") if i else Expr()
        result.append(pair + carry - 2 * pc)
        carry = ab + pc  # these cases are disjoint
    return result


def build(program: bytes, target: Target | None = None) -> ByteCore:
    """Construct one fixed-program, token-stepping CPU graph."""
    c = ByteCore(program, target)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)
    pc_oh = c.onehot_of_bits(c.pc, "pc", 256)

    def opcode_flag(op: int) -> Expr:
        return sum((pc_oh[i] for i, byte in enumerate(program) if byte == op), Expr())

    flags = {op: opcode_flag(op) for op in OPCODES}
    operand = [sum((pc_oh[i] for i in range(len(program) - 1)
                    if program[i + 1] >> bit & 1), Expr()) for bit in range(8)]
    fallthrough = [sum((pc_oh[i] for i in range(len(program))
                        if ((i + (1 if program[i] in (0, 255) else 2)) >> bit) & 1),
                       Expr()) for bit in range(8)]

    active = one - c.halted
    ldi, add, sta, lda, jnz, halt = (c.and_([active, flags[op]], f"is.{op:02x}")
                                      for op in (0x10, 0x20, 0x30, 0x40, 0x50, 0xFF))
    address = c.onehot_of_bits(operand, "ram.addr", RAM_SIZE)
    write = [c.and_([sta, x], f"ram.write.{i}") for i, x in enumerate(address)]
    read = [c.and_([lda, x], f"ram.read.{i}") for i, x in enumerate(address)]
    ram = c.slot_head("ram", dict(enumerate(write)), dict(enumerate(read)),
                      {2 + i: b for i, b in enumerate(c.acc)}, sta)

    summed = _add8(c, c.acc, operand)
    default_acc = one - ldi - add - lda
    next_acc = c.select_bits([([ldi], operand), ([add], summed), ([lda], ram),
                              ([default_acc], c.acc)], 8, "next.acc")
    acc_zero = c.and_([one - b for b in c.acc], "acc.zero")
    branch = c.and_([jnz, one - acc_zero], "branch")
    hold = c.halted + halt
    normal = one - branch - hold
    next_pc = c.select_bits([([branch], operand), ([hold], c.pc),
                             ([normal], fallthrough)], 8, "next.pc")
    next_halted = hold
    next_bits = [*next_pc, *next_acc, next_halted]

    # Replace the input token's output code with the next state's code.
    # The emitter clears all other circuit dims at this exit sublayer.
    floor = max(n.level for n in c.nodes)
    for i, (old, bit) in enumerate(zip(c.code, next_bits, strict=True)):
        c.ffn([(2 * KAPPA, bit, one), (-KAPPA, one, one)],
              f"exit.code{i}", into=old, min_level=floor)
    return c


def build_standalone(program: bytes, *, dtype=None):
    """Make a stock Qwen3_5ForCausalLM with this CPU in its weights.

    The 131,072 state tokens cover every PC/accumulator/halt combination with
    PC < 256. This builds a large embedding and output head, so a GPU machine
    is recommended. No pretrained language-model weights are involved.
    """
    import torch

    from chipformer.q35.tiny import build as build_qwen

    circuit = build(program)
    code_dims = [next(iter(e.terms)) for e in circuit.code]
    model = build_qwen(circuit, [*range(STATE_VOCAB), 0], code_dims,
                       dtype=torch.bfloat16 if dtype is None else dtype)
    with torch.no_grad():
        model.model.embed_tokens.weight[STATE_VOCAB].zero_()
    return model, circuit
