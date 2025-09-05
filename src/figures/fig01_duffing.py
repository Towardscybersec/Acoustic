"""Minimal deterministic "figure" used for the tests.

To keep the repository lightweight and avoid non-standard dependencies such as
``matplotlib`` we simply generate a small PGM (portable gray map) image
containing a sine-wave pattern.  The file format is plain ASCII and therefore
easy to hash for regression testing.
"""
from __future__ import annotations

import math
from pathlib import Path

from ..common.seeding import seed_all
from ..utils.paths import FIGURES_DIR


def run() -> Path:
    """Generate the example figure and return the path to the created file."""

    seed_all(0)
    width, height = 200, 100
    data = []
    for y in range(height):
        row = []
        for x in range(width):
            val = (math.sin(2 * math.pi * 5 * x / width) + 1.0) / 2.0
            row.append(int(val * 255))
        data.append(row)

    out = FIGURES_DIR / "fig01_test.pgm"
    with out.open("w", encoding="ascii") as f:
        f.write(f"P2\n{width} {height}\n255\n")
        for row in data:
            f.write(" ".join(str(v) for v in row) + "\n")
    return out


__all__ = ["run"]
