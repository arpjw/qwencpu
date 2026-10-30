"""A reusable byte CPU circuit whose program bytes arrive as input tokens.

The host supplies LOAD(address, byte) tokens, then the initial state token.
The model generates alternating TICK and state tokens. Four Gated DeltaNet
heads hold the 256-byte ROM, two more hold registers and RAM. The ISA logic
and ROM addressing are fixed weights; the program contents are recurrent
state, so one checkpoint can execute different programs on separate caches.

Only valid, halting programs are in scope. Unloaded ROM rows read as zero and
the reference emulator remains responsible for malformed-program errors.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from gdn_cpu.qwen8 import CODE_AMP, KAPPA, OPCODES, STATE_VOCAB, _add8
from gdn_cpu.reference import CPUState, RAM_SIZE


TICK = STATE_VOCAB
PAD = STATE_VOCAB + 1
LOAD_BASE = STATE_VOCAB + 2
ROM_SIZE = 256
VOCAB = LOAD_BASE + ROM_SIZE * 256


def load_token(address: int, value: int, *, rom_size: int = ROM_SIZE) -> int:
    if not 0 <= address < rom_size or not 0 <= value <= 255:
        raise ValueError("LOAD requires byte address and value")
    return LOAD_BASE + address * 256 + value


def program_tokens(program: bytes, *, rom_size: int = ROM_SIZE) -> list[int]:
    if not program or len(program) > rom_size:
        raise ValueError(f"program length must be 1..{rom_size} bytes")
    return [load_token(i, value, rom_size=rom_size) for i, value in enumerate(program)]


class DynamicByteCore(Circ):
    def __init__(self, target: Target | None = None, *, rom_size: int = ROM_SIZE) -> None:
        if not 1 <= rom_size <= ROM_SIZE:
            raise ValueError("rom_size must be 1..256")
        super().__init__(target or Target())
        self.rom_size = rom_size
        self.vocab = LOAD_BASE + rom_size * 256
        f = self.token_feature
        self.is_state = f("is_state", lambda t: int(t < STATE_VOCAB), full=True)
        self.is_tick = f("is_tick", lambda t: int(t == TICK), full=True)
        self.is_load = f("is_load", lambda t: int(LOAD_BASE <= t < self.vocab), full=True)
        self.raw_state = [f(f"state.b{i}", lambda t, i=i: (t >> i & 1)
                            if t < STATE_VOCAB else 0, full=True) for i in range(17)]
        self.load_address = [f(f"load.addr.b{i}",
                               lambda t, i=i: ((t - LOAD_BASE) >> (8 + i) & 1)
                               if LOAD_BASE <= t < self.vocab else 0, full=True)
                             for i in range(8)]
        self.load_data = [f(f"load.data.b{i}",
                            lambda t, i=i: ((t - LOAD_BASE) >> i & 1)
                            if LOAD_BASE <= t < self.vocab else 0, full=True)
                          for i in range(8)]
        self.code = [f(f"code.b{i}", lambda t, i=i:
                       CODE_AMP * (2 * ((t >> i) & 1) - 1), full=True)
                     for i in range(18)]
        self.keep = {next(iter(e.terms)) for e in self.code}


def _gated_onehot(c: DynamicByteCore, bits: list[Expr], gate: Expr,
                  name: str) -> list[Expr]:
    raw = c.onehot_of_bits(bits, name + ".raw", c.rom_size)
    return [c.and_([gate, bit], f"{name}[{i}]") for i, bit in enumerate(raw)]


def _rom_read(c: DynamicByteCore, name: str, write: list[Expr],
              query: list[Expr]) -> list[Expr]:
    """Split ROM key rows across at most two independent GDN heads."""
    chunks = []
    for bank in range((c.rom_size + 127) // 128):
        start = bank * 128
        rows = min(128, c.rom_size - start)
        keys = {i: write[start + i] for i in range(rows)}
        queries = {i: query[start + i] for i in range(rows)}
        chunks.append(c.slot_head(f"{name}.{bank}", keys, queries,
                                  {2 + i: bit for i, bit in enumerate(c.load_data)},
                                  c.is_load))
    return [sum((chunk[i] for chunk in chunks), Expr()) for i in range(8)]


def build(target: Target | None = None, *, rom_size: int = ROM_SIZE) -> DynamicByteCore:
    c = DynamicByteCore(target, rom_size=rom_size)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)

    reg = c.slot_head("registers", {0: c.is_state}, {0: c.is_tick},
                      {2 + i: bit for i, bit in enumerate(c.raw_state)}, c.is_state)
    pc, acc, halted = reg[:8], reg[8:16], reg[16]

    write = _gated_onehot(c, c.load_address, c.is_load, "rom.write")
    op_query = _gated_onehot(c, pc, c.is_tick, "rom.op.query")
    pc_plus_one = _add8(c, pc, [one, *([Expr()] * 7)])
    arg_query = _gated_onehot(c, pc_plus_one, c.is_tick, "rom.arg.query")
    opcode = _rom_read(c, "rom.op", write, op_query)
    operand = _rom_read(c, "rom.arg", write, arg_query)

    flags = {op: c.and_([bit if op >> i & 1 else one - bit
                        for i, bit in enumerate(opcode)], f"is.op.{op:02x}")
             for op in OPCODES}
    active = c.and_([c.is_tick, one - halted], "active")
    ldi, add, sta, lda, jnz, halt = (c.and_([active, flags[op]], f"is.{op:02x}")
                                      for op in (0x10, 0x20, 0x30, 0x40, 0x50, 0xFF))

    address = c.onehot_of_bits(operand, "ram.addr", RAM_SIZE)
    ram_write = [c.and_([sta, x], f"ram.write.{i}") for i, x in enumerate(address)]
    ram_read = [c.and_([lda, x], f"ram.read.{i}") for i, x in enumerate(address)]
    ram = c.slot_head("ram", dict(enumerate(ram_write)), dict(enumerate(ram_read)),
                      {2 + i: b for i, b in enumerate(acc)}, sta)

    summed = _add8(c, acc, operand)
    default_acc = one - ldi - add - lda
    next_acc = c.select_bits([([ldi], operand), ([add], summed), ([lda], ram),
                              ([default_acc], acc)], 8, "next.acc")
    acc_zero = c.and_([one - b for b in acc], "acc.zero")
    branch = c.and_([jnz, one - acc_zero], "branch")
    hold = halted + halt
    normal = one - branch - hold
    single = flags[0x00] + flags[0xFF]
    fallthrough = _add8(c, pc, [single, one - single, *([Expr()] * 6)])
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


def build_standalone(*, dtype=None, rom_size: int = ROM_SIZE):
    import torch

    from chipformer.q35.tiny import build as build_qwen

    circuit = build(rom_size=rom_size)
    code_dims = [next(iter(e.terms)) for e in circuit.code]
    model = build_qwen(circuit, list(range(circuit.vocab)), code_dims,
                       dtype=torch.bfloat16 if dtype is None else dtype)
    with torch.no_grad():
        model.model.embed_tokens.weight[PAD].zero_()
    return model, circuit
