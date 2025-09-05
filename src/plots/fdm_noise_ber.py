#!/usr/bin/env python3
"""BPSK BER simulation with RRC pulse shaping."""

import csv
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

from ..utils.paths import RESULTS_DIR, FIGURES_DIR


def run(short: bool = False) -> None:
    """Simulate BER curves and save plots/CSV."""
    rng = np.random.default_rng(2025)

    # -----------------------------
    # Root Raised Cosine filter (numerically stable)
    # -----------------------------
    def rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
        """Root raised cosine filter."""
        if beta < 0 or beta > 1:
            raise ValueError("beta must be in [0, 1]")
        N = span * sps
        t = np.arange(-N / 2, N / 2 + 1) / sps
        h = np.zeros_like(t, dtype=float)

        if beta == 0.0:
            beta = 1e-8

        for i, ti in enumerate(t):
            at0 = abs(ti) < 1e-12
            at_pm = abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12

            if at0:
                h[i] = 1.0 - beta + 4.0 * beta / np.pi
            elif at_pm:
                h[i] = (beta / np.sqrt(2.0)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            else:
                num = (
                    np.sin(np.pi * ti * (1.0 - beta))
                    + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
                )
                den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
                h[i] = num / den

        h /= np.sqrt(np.sum(h ** 2))
        return h

    # -----------------------------
    # Simulation parameters
    # -----------------------------
    Nbits = 20000 if not short else 2000
    sps = 8
    beta = 0.25
    span = 10
    Rs = 1_000.0
    fs = Rs * sps

    h = rrc_filter(beta, span, sps)

    bits = rng.integers(0, 2, size=Nbits)
    syms = 2 * bits - 1

    x_ups = np.zeros(Nbits * sps)
    x_ups[::sps] = syms
    tx = np.convolve(x_ups, h, mode="same")

    def awgn(sig: np.ndarray, snr_db: float) -> np.ndarray:
        p = np.mean(sig ** 2)
        snr = 10 ** (snr_db / 10.0)
        nvar = p / snr
        n = rng.normal(0.0, np.sqrt(nvar), size=len(sig))
        return sig + n

    def ber_from_rx(rx: np.ndarray) -> float:
        z = np.convolve(rx, h, mode="same")
        y = z[::sps]
        bhat = (y >= 0).astype(int)
        return float(np.mean(bhat != bits))

    SNR_dBs = np.linspace(-2, 16, 10)

    rows_awgn = []
    for snr in SNR_dBs:
        rx = awgn(tx, snr)
        rows_awgn.append([float(snr), ber_from_rx(rx)])

    amp_noise_std = 0.01
    phase_noise_std = 0.002
    fc = 12_000.0
    t = np.arange(len(tx)) / fs

    def impair_upconvert(sig: np.ndarray) -> np.ndarray:
        amp = 1.0 + rng.normal(0.0, amp_noise_std, size=len(sig))
        phi = rng.normal(0.0, phase_noise_std, size=len(sig))
        return sig * amp * np.cos(2 * np.pi * fc * t + phi)

    def downconvert(sig_c: np.ndarray) -> np.ndarray:
        lo = np.cos(2 * np.pi * fc * t)
        base = sig_c * lo * 2.0
        return base

    rows_imp = []
    for snr in SNR_dBs:
        tx_c = impair_upconvert(tx)
        rx_c = awgn(tx_c, snr)
        base = downconvert(rx_c)
        rows_imp.append([float(snr), ber_from_rx(base)])

    with open(RESULTS_DIR / "ber_awgn.csv", "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["SNR_dB", "BER"])
        cw.writerows(rows_awgn)

    with open(RESULTS_DIR / "ber_impaired.csv", "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["SNR_dB", "BER"])
        cw.writerows(rows_imp)

    def Q(x: float) -> float:
        return 0.5 * erfc(x / sqrt(2.0))

    theory = [Q(np.sqrt(2.0 * 10 ** (e / 10.0))) for e in SNR_dBs]

    plt.figure(figsize=(6, 4))
    plt.semilogy(SNR_dBs, [r[1] for r in rows_awgn], marker="o", label="AWGN (sim)")
    plt.semilogy(SNR_dBs, [r[1] for r in rows_imp], marker="s", label="Impaired (sim)")
    plt.semilogy(SNR_dBs, theory, linestyle="--", label="BPSK AWGN (theory)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BPSK BER with RRC pulse shaping")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ber_rrc_bpsk.png", dpi=200)

    print("[OK] Saved BER curves and plot.")


if __name__ == "__main__":
    run()
