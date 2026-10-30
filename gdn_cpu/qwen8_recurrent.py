"""A byte CPU variant with registers and RAM in Gated DeltaNet state.

Generation alternates a state token and a TICK token. A state token commits
PC/accumulator/halt to one recurrent head; TICK reads that head and predicts
the next state token. The model predicts both token kinds, so the host only
supplies an initial state and a fixed compiled program.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from gdn_cpu.qwen8 import CODE_AMP, KAPPA, OPCODES, STATE_VOCAB, _add8
from gdn_cpu.reference import MAX_PROGRAM_SIZE, RAM_SIZE


TICK = STATE_VOCAB
PAD = STATE_VOCAB + 1


class RecurrentByteCore(Circ):
    def __init__(self, program: bytes, target: Target | None = None) -> None:
        if not program or len(program) > MAX_PROGRAM_SIZE:
            raise ValueError("program length must be 1..256 bytes")
        super().__init__(target or Target())
        self.program = bytes(program)
        f = self.token_feature
        self.is_state = f("is_state", lambda t: int(t < STATE_VOCAB), full=True)
        self.is_tick = f("is_tick", lambda t: int(t == TICK), full=True)
        raw = [f(f"state.b{i}", lambda t, i=i: (t >> i & 1)
                 if t < STATE_VOCAB else 0, full=True) for i in range(17)]
        self.code = [f(f"code.b{i}", lambda t, i=i:
                       CODE_AMP * (2 * ((t >> i) & 1) - 1), full=True)
                     for i in range(18)]
        self.keep = {next(iter(e.terms)) for e in self.code}
        self.raw_state = raw


def build(program: bytes, target: Target | None = None) -> RecurrentByteCore:
    c = RecurrentByteCore(program, target)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)

    # Query only on TICK. Its token contains no state bits, so this head is the
    # sole source of the machine registers at the instruction boundary.
    reg = c.slot_head("registers", {0: c.is_state}, {0: c.is_tick},
                      {2 + i: bit for i, bit in enumerate(c.raw_state)}, c.is_state)
    pc, acc, halted = reg[:8], reg[8:16], reg[16]
    pc_oh = c.onehot_of_bits(pc, "pc", 256)

    def flag(op: int) -> Expr:
        return sum((pc_oh[i] for i, byte in enumerate(program) if byte == op), Expr())

    flags = {op: flag(op) for op in OPCODES}
    operand = [sum((pc_oh[i] for i in range(len(program) - 1)
                    if program[i + 1] >> bit & 1), Expr()) for bit in range(8)]
    fallthrough = [sum((pc_oh[i] for i in range(len(program))
                        if ((i + (1 if program[i] in (0, 255) else 2)) >> bit) & 1),
                       Expr()) for bit in range(8)]
    active = c.and_([c.is_tick, one - halted], "active")
    ldi, add, sta, lda, jnz, halt = (c.and_([active, flags[op]], f"is.{op:02x}")
                                      for op in (0x10, 0x20, 0x30, 0x40, 0x50, 0xFF))
    address = c.onehot_of_bits(operand, "ram.addr", RAM_SIZE)
    write = [c.and_([sta, x], f"ram.write.{i}") for i, x in enumerate(address)]
    read = [c.and_([lda, x], f"ram.read.{i}") for i, x in enumerate(address)]
    ram = c.slot_head("ram", dict(enumerate(write)), dict(enumerate(read)),
                      {2 + i: b for i, b in enumerate(acc)}, sta)

    summed = _add8(c, acc, operand)
    default_acc = one - ldi - add - lda
    next_acc = c.select_bits([([ldi], operand), ([add], summed), ([lda], ram),
                              ([default_acc], acc)], 8, "next.acc")
    acc_zero = c.and_([one - b for b in acc], "acc.zero")
    branch = c.and_([jnz, one - acc_zero], "branch")
    hold = halted + halt
    normal = one - branch - hold
    next_pc = c.select_bits([([branch], operand), ([hold], pc),
                             ([normal], fallthrough)], 8, "next.pc")
    next_state = [*next_pc, *next_acc, hold]

    floor = max(n.level for n in c.nodes)
    for i, code in enumerate(c.code):
        if i < 17:
            terms = c.t_const(-KAPPA) + c.t_and([c.is_tick, next_state[i]], 2 * KAPPA)
        else:
            terms = c.t_const(-KAPPA) + c.t_and([c.is_state], 2 * KAPPA)
        c.ffn(terms, f"exit.code{i}", into=code, min_level=floor)
    return c


def build_standalone(program: bytes, *, dtype=None):
    import torch

    from chipformer.q35.tiny import build as build_qwen

    circuit = build(program)
    code_dims = [next(iter(e.terms)) for e in circuit.code]
    model = build_qwen(circuit, [*range(STATE_VOCAB), TICK, 0], code_dims,
                       dtype=torch.bfloat16 if dtype is None else dtype)
    with torch.no_grad():
        model.model.embed_tokens.weight[PAD].zero_()
    return model, circuit
