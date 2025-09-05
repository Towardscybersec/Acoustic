"""Utilities for deterministic seeding across libraries."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

# ``numpy`` is optional; many of the unit tests operate purely on Python lists
# and we do not want to make the whole package fail to import if the dependency
# is missing.  The small helper simply skips seeding of ``numpy`` when it is not
# installed which mirrors the behaviour of the real project where some modules
# are optional.
try:  # pragma: no cover - exercised indirectly in tests
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy not available
    np = None

try:  # Optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may not be installed
    torch = None


@dataclass(frozen=True)
class SeedBundle:
    """Container storing the seed used for each library."""

    python: int
    numpy: int
    torch: Optional[int] = None


def seed_all(seed: int) -> SeedBundle:
    """Seed Python's ``random`` module, :mod:`numpy` and :mod:`torch` if
    available.

    Parameters
    ----------
    seed:
        Base integer seed.  The same value is used for all libraries to ease
        reproducibility.

    Returns
    -------
    SeedBundle
        Dataclass containing the seeds that were applied.  The information can
        be serialised to JSON to record the experimental conditions.
    """

    random.seed(seed)
    if np is not None:  # pragma: no branch - depends on optional import
        np.random.seed(seed)

    torch_seed: Optional[int] = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU not in tests
            torch.cuda.manual_seed_all(seed)
        torch_seed = seed

    # Also propagate to the OS level for libraries that rely on it
    os.environ["PYTHONHASHSEED"] = str(seed)

    return SeedBundle(python=seed, numpy=seed if np is not None else None, torch=torch_seed)


__all__ = ["SeedBundle", "seed_all"]
