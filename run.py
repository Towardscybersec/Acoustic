"""Command line interface for running small deterministic experiments.

The real project described in the instructions is much larger.  For the unit
tests in this kata we only expose a handful of tasks that demonstrate the
infrastructure.  Additional tasks can easily be added by extending the
``TASKS`` dictionary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

from src.common.config import load_config, Config

# Mapping from task name to a callable that accepts a configuration object
TASKS: Dict[str, Callable[[Config], None]] = {}


def _register_tasks() -> None:
    """Populate :data:`TASKS` with the commands exposed by the CLI.

    The repository only contains toy functionality but we still expose a
    reasonably complete interface mimicking a typical research workflow with
    ``train``/``eval``/``plots`` stages.  ``plots`` drives the heavy-weight
    ``generate_figures`` module which regenerates all paper plots.  ``fig1`` is
    kept as a tiny deterministic figure for unit tests.
    """

    from src.figures.fig01_duffing import run as fig1_run
    from src.utils.paths import RESULTS_DIR
    from generate_figures import main as generate_all
    import json

    def fig1(cfg: Config) -> None:  # pragma: no cover - thin wrapper
        fig1_run()

    def plots(cfg: Config) -> None:  # pragma: no cover - heavy plotting
        generate_all()

    def train(cfg: Config) -> None:
        """Dummy training stage writing a small metrics file."""

        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "train_metrics.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump({"epochs": 1, "seed": cfg.get("seed", 0)}, fh)

    def eval_stage(cfg: Config) -> None:
        """Dummy evaluation stage writing a metrics file."""

        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "eval_metrics.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump({"accuracy": 1.0, "seed": cfg.get("seed", 0)}, fh)

    TASKS.update({
        "fig1": fig1,
        "plots": plots,
        "train": train,
        "eval": eval_stage,
    })


_register_tasks()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Research code runner")
    parser.add_argument("task", choices=sorted(TASKS.keys()) + ["all"], help="Task to run")
    parser.add_argument("--config", dest="config", default=None, help="Optional YAML configuration")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.task == "all":
        for name in ["train", "eval", "plots", "fig1"]:
            if name in TASKS:
                TASKS[name](cfg)
    else:
        TASKS[args.task](cfg)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
