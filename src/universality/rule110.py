"""Implementation of the Rule 110 cellular automaton.

The original research code used NumPy for efficiency.  To keep the exercises
self‑contained and avoid heavy dependencies we provide a small pure Python
implementation that mirrors the behaviour.  The update rule is

``U(a, b, c) = b + c - b*c - a*b*c``.

The :func:`evolve` function returns the full history as a list of lists so that
it works in environments without NumPy.  When NumPy is available the history is
converted to a :class:`numpy.ndarray` for convenience.
"""
from __future__ import annotations

from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


def _update_triplet(a: int, b: int, c: int) -> int:
    return b + c - b * c - a * b * c


def evolve(initial: Iterable[int], steps: int):
    """Evolve ``initial`` for ``steps`` time steps."""

    state = [int(v) for v in initial]
    width = len(state)
    history: List[List[int]] = [state.copy()]

    for _ in range(steps):
        new_state = [0] * width
        for i in range(width):
            a = state[(i - 1) % width]
            b = state[i]
            c = state[(i + 1) % width]
            new_state[i] = _update_triplet(a, b, c)
        state = new_state
        history.append(state.copy())

    if np is not None:  # pragma: no cover - exercised when numpy present
        return np.array(history, dtype=int)
    return history


__all__ = ["evolve"]
