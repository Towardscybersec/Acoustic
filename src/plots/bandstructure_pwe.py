"""Plane-wave expansion band-structure computation."""

import csv
import json
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import j1
import pandas as pd

from ..utils.paths import RESULTS_DIR, FIGURES_DIR


def run(short: bool = False) -> None:
    """Compute PnC band structure and save metrics/plot."""
    # Lattice/material parameters - higher contrast and larger fill for a wider gap
    a = 1e-3
    r = 0.42 * a
    c_host = 1500.0
    c_inc = 250.0
    G_cut = 4  # slightly richer basis

    # Path Γ–X–M–Γ
    def k_point(lbl):
        if lbl == "G":
            return np.array([0.0, 0.0])
        if lbl == "X":
            return np.array([np.pi / a, 0.0])
        if lbl == "M":
            return np.array([np.pi / a, np.pi / a])
        raise ValueError(lbl)

    path = ["G", "X", "M", "G"]
    Nk = 25

    b1 = np.array([2 * np.pi / a, 0.0])
    b2 = np.array([0.0, 2 * np.pi / a])
    Gs = []
    for nx in range(-G_cut, G_cut + 1):
        for ny in range(-G_cut, G_cut + 1):
            Gs.append(nx * b1 + ny * b2)
    Gs = np.array(Gs)
    NG = len(Gs)

    s_host = 1.0 / (c_host ** 2)
    s_inc = 1.0 / (c_inc ** 2)
    from math import pi

    Acell = a * a
    fill = pi * r * r / Acell

    def chi_of_G(G):
        g = np.linalg.norm(G)
        if g < 1e-12:
            return fill
        x = g * r
        return 2.0 * fill * j1(x) / x

    def build_S():
        S = np.zeros((NG, NG), dtype=np.complex128)
        cache = {}
        for i in range(NG):
            for j in range(NG):
                dG = tuple((Gs[i] - Gs[j]).tolist())
                if dG not in cache:
                    cache[dG] = chi_of_G(np.array(dG))
                chi = cache[dG]
                S[i, j] = (s_host if i == j else 0.0) + (s_inc - s_host) * chi
        return S

    S = build_S()

    # Build k-path
    ks = []
    ticks = []
    kx = [0.0]
    for seg in range(len(path) - 1):
        k0, k1 = k_point(path[seg]), k_point(path[seg + 1])
        for t in range(Nk):
            s = t / (Nk - 1)
            k = (1 - s) * k0 + s * k1
            if len(ks) > 0:
                kx.append(kx[-1] + np.linalg.norm(k - ks[-1]))
            ks.append(k)
        ticks.append((kx[-Nk + 1], path[seg]))
    ticks.append((kx[-1], path[-1]))

    # Solve and collect first 10 bands
    rows = []
    for i, k in enumerate(ks):
        A = np.diag([np.dot(k + G, k + G) for G in Gs]).astype(np.complex128)
        w2, _ = la.eig(A, S)
        w2 = np.real(w2)
        w2[w2 < 0] = np.nan
        f = np.sqrt(w2) / (2 * np.pi)
        norm = f * a / c_host
        order = np.argsort(norm)
        NB = min(10, len(order))
        for b in range(NB):
            rows.append([i, b + 1, float(f[order[b]]), float(norm[order[b]])])

    # Save CSV
    csv_path = RESULTS_DIR / "bands_pwe_improved.csv"
    with open(csv_path, "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["k_index", "band", "freq_hz", "norm_a_over_lambda"])
        cw.writerows(rows)

    # Compute global bandgap between band1 and band2
    df = pd.DataFrame(rows, columns=["k_index", "band", "freq_hz", "norm_a_over_lambda"])
    b1 = df[df.band == 1].sort_values("k_index")["norm_a_over_lambda"]
    b2 = df[df.band == 2].sort_values("k_index")["norm_a_over_lambda"]
    gap = float(b2.min() - b1.max())
    mid = 0.5 * (float(b2.min()) + float(b1.max()))
    gap_ratio = float(gap / mid) if mid > 0 else float("nan")
    with open(RESULTS_DIR / "bands_pwe_gap_metrics.json", "w") as f:
        json.dump({"global_gap_norm": gap, "gap_ratio": gap_ratio}, f, indent=2)

    # Plot
    plt.figure(figsize=(7, 4))
    NB = int(df["band"].max())
    for b in range(1, NB + 1):
        mask = df.band == b
        xs = df[mask]["k_index"].values
        ys = df[mask]["norm_a_over_lambda"].values
        plt.plot(xs, ys)
    for x, lbl in ticks:
        plt.axvline(x, linestyle="--", linewidth=0.8)
    plt.xticks([t[0] for t in ticks], [t[1] for t in ticks])
    plt.ylabel("a/λ = f·a/c_host")
    plt.title("PnC band structure (wider gap). Δ/ω_mid = {:.3f}".format(gap_ratio))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bands_pwe_improved.png", dpi=200)
    print("[OK] Saved band structure and gap metrics.")


if __name__ == "__main__":
    run()
