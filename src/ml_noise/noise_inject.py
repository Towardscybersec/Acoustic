"""Utilities for injecting quantisation noise based on ENOB."""
from __future__ import annotations

import random
from typing import Iterable, Optional

import math

try:  # ``numpy`` is optional in the execution environment
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy not installed
    np = None

from ..common.seeding import seed_all


def enob_to_noise_std(enob: float, full_scale: float = 1.0) -> float:
    """Convert ENOB to an equivalent noise standard deviation.

    The relationship is derived from the quantisation noise model where the RMS
    quantisation error is ``lsb/\sqrt{12}`` and ``lsb = full_scale / 2**enob``.
    """

    step = full_scale / (2 ** enob)
    return step / math.sqrt(12)


def noise_std_to_enob(std: float, full_scale: float = 1.0) -> float:
    """Inverse of :func:`enob_to_noise_std`."""

    step = std * math.sqrt(12)
    return math.log2(full_scale / step)


def inject_noise(x: Iterable[float], enob: float, seed: Optional[int] = None):
    """Add uniform quantisation-like noise to ``x``.

    ``x`` can be either a sequence of floats or a NumPy array.  The return type
    matches the input type, keeping the helper convenient for unit tests without
    imposing a hard dependency on NumPy.
    """

    if seed is not None:
        seed_all(seed)

    lsb = enob_to_noise_std(enob) * (12 ** 0.5)  # convert back to LSB magnitude
    if np is not None and isinstance(x, np.ndarray):
        noise = np.random.uniform(-0.5, 0.5, size=x.shape) * lsb
        return x + noise
    else:
        return [float(v) + (random.random() - 0.5) * lsb for v in x]


__all__ = ["enob_to_noise_std", "noise_std_to_enob", "inject_noise"]
