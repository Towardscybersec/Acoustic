"""1D SSH model illustrating topological edge modes."""

import json
import numpy as np
import matplotlib.pyplot as plt

from ..utils.paths import RESULTS_DIR, FIGURES_DIR


def run(short: bool = False) -> None:
    """Generate SSH chain metrics and mode plots."""
    rng = np.random.default_rng(2025)
    N = 80
    m = 1.0
    k1 = 1.0
    k2 = 3.0
    disorder = 0.05
    K = np.zeros((N, N))

    def add(i, j, val):
        if 0 <= i < N and 0 <= j < N:
            K[i, j] += val

    for i in range(N - 1):
        k = k1 if (i % 2 == 0) else k2
        k *= (1.0 + rng.uniform(-disorder, disorder))
        add(i, i, k)
        add(i + 1, i + 1, k)
        add(i, i + 1, -k)
        add(i + 1, i, -k)

    w2, V = np.linalg.eigh(K / m)
    w = np.sqrt(np.maximum(0.0, w2))
    idx = np.argsort(w)
    w = w[idx]
    V = V[:, idx]

    def ipr(vec):
        num = np.sum(vec ** 4)
        den = (np.sum(vec ** 2)) ** 2
        return num / den

    edge_len = 10
    metrics = []
    for i in range(6):
        vec = V[:, i]
        efrac = float(np.sum(vec[:edge_len] ** 2) + np.sum(vec[-edge_len:] ** 2))
        metrics.append({"mode": int(i), "omega": float(w[i]), "IPR": float(ipr(vec)), "edge_energy_frac": efrac})

    with open(RESULTS_DIR / "ssh_metrics.json", "w") as jf:
        json.dump(metrics, jf, indent=2)

    plt.figure(figsize=(7, 5))
    for n in range(6):
        plt.subplot(3, 2, n + 1)
        plt.plot(V[:, n])
        plt.title(f"Mode {n}  ω={w[n]:.3f}")
        plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ssh_modes_improved.png", dpi=200)
    print("[OK] Saved SSH metrics and plot.")


if __name__ == "__main__":
    run()
