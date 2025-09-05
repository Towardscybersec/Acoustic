"""2D FDTD line-defect waveguide with transmission sweep S21(f)."""

import csv
import numpy as np
import matplotlib.pyplot as plt

from ..utils.paths import RESULTS_DIR, FIGURES_DIR


def run(short: bool = False) -> None:
    """Run the FDTD simulation and save snapshot and S21 plot."""
    Nx, Ny = (220, 140) if not short else (60, 40)
    dx = dy = 1e-3
    c0 = 1500.0
    dt = 0.5 * dx / c0
    steps = 2800 if not short else 400
    eta = 5.0

    c = c0 * np.ones((Ny, Nx), dtype=float)

    a = 12
    rad = 0.40 * a
    x0, x1 = 40, 180 if not short else (10, 50)
    y_mid = Ny // 2
    slab_half = 40 if not short else 10
    defect_row = y_mid

    Y, X = np.ogrid[:Ny, :Nx]
    for ix in range(x0, x1, a):
        for iy in range(y_mid - slab_half, y_mid + slab_half, a):
            if iy == defect_row:
                continue
            mask = (X - ix) ** 2 + (Y - iy) ** 2 <= (rad ** 2)
            c[mask] = 340.0

    sponge = np.ones((Ny, Nx), dtype=float)
    pad = 12 if not short else 6
    for i in range(Nx):
        for j in range(Ny):
            sx = max(0, pad - min(i, Nx - 1 - i)) / pad
            sy = max(0, pad - min(j, Ny - 1 - j)) / pad
            sponge[j, i] = 1.0 + 8.0 * (sx + sy)

    src_x, src_y = (20, defect_row)
    rec_x = Nx - 30 if not short else Nx - 10
    window = 3

    def run_single(f0: float):
        p = np.zeros((Ny, Nx), dtype=float)
        v = np.zeros_like(p)
        omega = 2 * np.pi * f0
        rec = []
        for n in range(steps):
            lap = (np.roll(p, -1, axis=1) - 2 * p + np.roll(p, 1, axis=1)) / dx ** 2 + \
                  (np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)) / dy ** 2
            a_tt = c ** 2 * lap - eta * v
            v += dt * a_tt
            p += dt * v
            v /= sponge
            p[src_y, src_x] += 0.1 * np.sin(omega * n * dt)
            rec.append(np.mean(p[src_y - window:src_y + window + 1, rec_x]))
        return np.array(rec), p

    f_probe = 12_000.0
    trace, snap = run_single(f_probe)

    plt.figure(figsize=(7, 3))
    plt.imshow(snap, origin="lower", aspect="auto", cmap="viridis")
    plt.title(f"FDTD snapshot at {f_probe:.1f} Hz")
    plt.colorbar(label="p")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fdtd_snapshot_improved.png", dpi=200)

    f_start, f_stop, f_step = 9_000.0, 15_000.0, 250.0
    freqs = np.arange(f_start, f_stop + f_step / 2, f_step)
    S21 = []
    for f0 in freqs:
        r, _ = run_single(f0)
        transient = int(0.4 * len(r))
        rms = float(np.sqrt(np.mean(r[transient:] ** 2)))
        S21.append(rms)

    with open(RESULTS_DIR / "fdtd_S21.csv", "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["freq_hz", "rms_transmission"])
        cw.writerows(zip(freqs, S21))

    plt.figure(figsize=(6, 3))
    plt.plot(freqs, S21, marker="o")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|S21| (RMS at receiver)")
    plt.title("Line-defect transmission S21(f)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fdtd_S21.png", dpi=200)

    print("[OK] Saved FDTD snapshot and S21(f).")


if __name__ == "__main__":
    run()
