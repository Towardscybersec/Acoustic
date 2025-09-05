"""Duffing oscillator comb example generating spectrum and Allan deviation."""

import json
import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ..utils.paths import RESULTS_DIR, FIGURES_DIR


def run(short: bool = False) -> None:
    """Generate Duffing comb metrics and plots.

    Args:
        short: unused placeholder for interface compatibility.
    """
    # Parameters tuned for visible sidebands
    f0 = 20_000.0
    omega0 = 2 * np.pi * f0
    zeta = 2e-3
    alpha = 5e15
    Famp = 3e-3
    fd = 0.999 * f0
    omegad = 2 * np.pi * fd
    T = 0.5
    fs = 2_000_000.0
    dt = 1.0 / fs
    N = int(T * fs)

    x = 0.0
    v = 0.0
    xs = np.zeros(N)
    for i in range(N):
        t = i * dt
        acc = -2 * zeta * omega0 * v - omega0 ** 2 * x - alpha * (x ** 3) + Famp * np.cos(omegad * t)
        v += acc * dt
        x += v * dt
        xs[i] = x

    # Spectrum
    win = np.hanning(N)
    X = rfft(xs * win)
    freqs = rfftfreq(N, dt)
    S = (np.abs(X) ** 2)
    S /= np.max(S)

    # Peak detection: detect up to 11 lines around the carrier and harmonics
    peaks, props = find_peaks(S, height=1e-6, distance=int(0.0004 / dt))
    peak_freqs = freqs[peaks]
    peak_amps = props["peak_heights"]

    def linewidth(f0, S, freqs):
        i0 = np.argmin(np.abs(freqs - f0))
        peak = S[i0]
        th = peak / 2.0
        i1 = i0
        while i1 > 0 and S[i1] > th:
            i1 -= 1
        i2 = i0
        while i2 < len(S) - 1 and S[i2] > th:
            i2 += 1
        return freqs[i2] - freqs[i1]

    # Sort and keep 9 most prominent lines near multiples of fd
    order = np.argsort(-peak_amps)
    lines = []
    for idx in order[:12]:
        f = float(peak_freqs[idx])
        a = float(peak_amps[idx])
        lw = float(linewidth(f, S, freqs))
        lines.append({"f_hz": f, "norm_height": a, "linewidth_hz": lw})

    # Allan deviation (simple zero-cross estimator)
    def allan_deviation(x, fs, f_carrier, taus):
        zeros = np.where(np.diff(np.sign(x)) > 0)[0]
        periods = np.diff(zeros) / fs
        inst_f = 1.0 / periods
        ad = []
        for tau in taus:
            m = max(1, int(round(tau * fs * 0.5)))
            if m * 2 >= len(inst_f):
                ad.append(np.nan)
                continue
            y = np.array([np.mean(inst_f[i * m:(i + 1) * m]) for i in range(len(inst_f) // m)])
            avar = 0.5 * np.mean((y[1:] - y[:-1]) ** 2)
            ad.append(np.sqrt(avar) / f_carrier)
        return np.array(ad)

    taus = np.logspace(-4, -1, 25)
    adev = allan_deviation(xs, fs, fd, taus)

    # Save
    with open(RESULTS_DIR / "duffing_comb_metrics_improved.json", "w") as jf:
        json.dump({"carrier_hz": fd, "num_lines_detected": len(lines), "lines": lines,
                   "allan_tau_s": list(map(float, taus)), "allan_dev": list(map(float, adev))}, jf, indent=2)

    # Plots
    plt.figure(figsize=(7, 3))
    plt.semilogy(freqs, S + 1e-18)
    plt.xlim(fd * 0.6, fd * 3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Norm. power")
    plt.title("Duffing comb with multiple sidebands")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duffing_spectrum_improved.png", dpi=200)

    plt.figure(figsize=(5, 3))
    plt.loglog(taus, np.where(np.isnan(adev), np.nan, adev))
    plt.xlabel("Averaging time τ (s)")
    plt.ylabel("Allan deviation (fractional)")
    plt.title("Allan deviation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "allan_deviation_improved.png", dpi=200)

    print("[OK] Saved Duffing comb metrics and plots.")


if __name__ == "__main__":
    run()
