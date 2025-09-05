"""Simple GISG grammar parser used in the tests.

The grammar recognised is intentionally tiny – each line contains an
instruction ``T1``, ``T2`` or ``T3`` followed by zero or more flag tokens
``F1`` – ``F3``.  Anything following a ``#`` character is treated as a
comment and ignored.

Example
-------
The following snippet illustrates the accepted format::

    T1 F1 F2
    T2
    T3 F3  # comment

which would be parsed into a list of three instructions with the
corresponding flag sets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict


@dataclass
class Instruction:
    op: str
    flags: List[str]

    def to_dict(self) -> Dict[str, List[str]]:  # pragma: no cover - trivial
        return {"op": self.op, "flags": list(self.flags)}


def parse(text: str) -> List[Instruction]:
    """Parse a GISG program from *text*.

    The grammar understood by the tests is intentionally tiny and very strict
    to keep the implementation self‑contained.  Only three opcodes are
    recognised: ``"T1"``, ``"T2"`` and ``"T3"``.  Each opcode may be followed by
    zero or more flag tokens chosen from ``"F1"``, ``"F2"`` and ``"F3"``.  Any
    other token results in a :class:`ValueError` which mirrors the behaviour of
    a real parser that would reject malformed programs.
    """

    allowed_ops = {"T1", "T2", "T3"}
    allowed_flags = {"F1", "F2", "F3"}

    instructions: List[Instruction] = []
    for raw_line in text.splitlines():
        # Remove comments and surrounding whitespace
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        tokens = line.split()
        op = tokens[0]
        if op not in allowed_ops:
            raise ValueError(f"Unknown instruction {op!r}")

        flags: List[str] = []
        for tok in tokens[1:]:
            if tok not in allowed_flags:
                raise ValueError(f"Unknown token {tok!r}")
            if tok in flags:
                raise ValueError(f"Duplicate flag {tok!r}")
            flags.append(tok)

        instructions.append(Instruction(op=op, flags=flags))

    return instructions


def emit_json_trace(instructions: Iterable[Instruction], path: str | Path) -> None:
    """Write a JSON trace of ``instructions`` to *path*.

    Each instruction is serialised to a dictionary with keys ``op`` and
    ``flags``.  The output is deterministic and easy to diff which is useful for
    unit tests and reproducible research artefacts.
    """

    serialised = [instr.to_dict() for instr in instructions]
    p = Path(path)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(serialised, fh, indent=2, sort_keys=True)


__all__ = ["Instruction", "parse", "emit_json_trace"]
