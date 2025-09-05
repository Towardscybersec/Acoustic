"""Lightweight configuration system based on YAML files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML might not be installed
    yaml = None


@dataclass
class Config:
    """Configuration container with attribute-style access."""

    data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple proxy
        try:
            return self.data[item]
        except KeyError as exc:  # pragma: no cover - error message is clear
            raise AttributeError(item) from exc


def load_config(path: Optional[str | Path]) -> Config:
    """Load configuration from a YAML file.

    If ``path`` is ``None`` or the file does not exist, an empty configuration is
    returned.  The function never raises when the file is missing which simplifies
    command line interfaces.
    """

    if path is None:
        return Config({})

    p = Path(path)
    if not p.exists() or yaml is None:
        return Config({})

    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


__all__ = ["Config", "load_config"]
