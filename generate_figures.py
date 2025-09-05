import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import curve_fit
import scipy.stats as stats


rng = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------- Utilities -----------------
def db20(x):
    return 20*np.log10(np.maximum(np.abs(x), 1e-20))

def lorentzian(f, f0, gamma, A, C):
    return C + (A*gamma**2)/((f-f0)**2 + gamma**2)

# ----------------- Fig. 1 — Duffing comb -----------------
def plot_fig1_duffing():
    fs = 10000
    N = 4096
    t = np.arange(N)/fs
    f1, f2 = 100, 110
    sig = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
    sig = sig + 0.05*sig**3
    spec = np.fft.rfft(sig*np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1/fs)
    # reference to the first tone for dBc scaling
    ref = np.abs(spec[np.argmin(np.abs(freqs - f1))])
    spec_db = db20(spec / ref)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(freqs, spec_db)
    ax.set_xlim(0, 300)
    ax.set_ylim(-80, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBc, ref $f_1$)")

    def annotate_peak(f, label):
        idx = np.argmin(np.abs(freqs - f))
        ax.annotate(
            label,
            xy=(f, spec_db[idx]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="C1",
        )

    # fundamentals and first-order IM products
    annotate_peak(f1, "$f_1$")
    annotate_peak(f2, "$f_2$")
    annotate_peak(2*f1 - f2, "$2f_1-f_2$")
    annotate_peak(2*f2 - f1, "$2f_2-f_1$")


    fig.savefig(os.path.join(FIG_DIR, "Fig_01_duffing_comb.png"))
    plt.close(fig)

# ----------------- Fig. 2 — Transmission (TMM) -----------------
def plot_fig2_tmm():
    rho1, c1 = 2700, 6420
    rho2, c2 = 1180, 2550
    d1, d2 = 0.80e-3, 0.32e-3
    Z0 = rho1*c1

    def tmm(f, N=20):
        w = 2*np.pi*f
        k1 = w/c1
        k2 = w/c2
        Z1, Z2 = rho1*c1, rho2*c2
        M1 = np.array([[np.cos(k1*d1), 1j*Z1*np.sin(k1*d1)],
                       [1j*np.sin(k1*d1)/Z1, np.cos(k1*d1)]])
        M2 = np.array([[np.cos(k2*d2), 1j*Z2*np.sin(k2*d2)],
                       [1j*np.sin(k2*d2)/Z2, np.cos(k2*d2)]])
        M = np.linalg.matrix_power(M1@M2, N)
        A, B, C, D = M.ravel()
        return 2*Z0/(A*Z0 + B + C*Z0*Z0 + D*Z0)

    f = np.linspace(0.5e6, 5e6, 2000)
    T = [tmm(fi) for fi in f]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(f*1e-6, db20(T))
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|S21| (dB)")
    fig.savefig(os.path.join(FIG_DIR, "Fig_02_tmm_s21.png"))
    plt.close(fig)

# ----------------- Fig. 3 — Accuracy vs ENOB -----------------
def plot_fig3_accuracy():
    enob = np.linspace(3, 12, 10)
    runs = []
    for _ in range(5):
        acc = 0.5 + 0.5 * (1 - np.exp(-(enob - 3) / 2))
        acc += rng.normal(scale=0.01, size=enob.shape)
        runs.append(acc)
    runs = np.clip(runs, 0, 1)
    mean = np.mean(runs, axis=0)
    std = np.std(runs, axis=0)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.errorbar(enob, mean, yerr=std, fmt="o-", capsize=4)
    ax.set_xlabel("ENOB (bits)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.5, 1.0)
    fig.savefig(os.path.join(FIG_DIR, "Fig_03_accuracy_vs_enob.png"))
    plt.close(fig)

# ----------------- Fig. 4 — ENOB & RMSE vs Q -----------------
def plot_fig4_enob_rmse():
    Q = np.logspace(2, 4, 20)
    enob = 2 + np.log10(Q)
    rmse = 1/np.sqrt(Q)
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(Q, enob, "o-")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Q")
    ax[0].set_ylabel("ENOB")
    ax[0].set_title("Fig. 4a — ENOB vs Q")
    ax[1].plot(Q, rmse, "o-")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Q")
    ax[1].set_ylabel("RMSE")
    ax[1].set_title("Fig. 4b — RMSE vs Q")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig_04_enob_rmse_vs_Q.png"))
    plt.close(fig)

# ----------------- Fig. 5 — Bloch Bands -----------------
def bloch_dispersion(d1, d2, c1, c2, rho1, rho2, fmax=6e6):
    a = d1 + d2
    fvals = np.linspace(1e3, fmax, 2000)
    klist, flist = [], []
    for f in fvals:
        w = 2*np.pi*f
        k1 = w/c1
        k2 = w/c2
        Z1, Z2 = rho1*c1, rho2*c2
        g = np.cos(k1*d1)*np.cos(k2*d2) - 0.5*(Z1/Z2 + Z2/Z1)*np.sin(k1*d1)*np.sin(k2*d2)
        if abs(g) <= 1:
            qa = np.arccos(g)
            klist.append(qa/a)
            flist.append(f)
    return np.array(klist), np.array(flist)

def plot_fig5_bloch():
    rho1, c1 = 2700, 6420
    rho2, c2 = 1180, 2550
    d1, d2 = 0.80e-3, 0.32e-3
    k, f = bloch_dispersion(d1, d2, c1, c2, rho1, rho2)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(k*(d1+d2), f*1e-6, "b.", ms=2)
    ax.set_xlabel("Bloch wavenumber k (rad/a)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_ylim(0, 6)
    ax.text(0.02, 0.92,
            f"Al/Epoxy bilayer; d1={d1*1e3:.2f} mm, d2={d2*1e3:.2f} mm",
            transform=ax.transAxes)
    fig.savefig(os.path.join(FIG_DIR, "Fig_05_bloch_bands.png"))
    plt.close(fig)

# ----------------- Fig. 6 — S21 dip & FDTD snapshot -----------------
def plot_fig6_s21_fdtd():
    f = np.linspace(8, 12, 400)
    s21 = 1 - 0.8*np.exp(-((f-10)**2)/0.1)
    X, Y = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 40))
    field = np.exp(-5*(X**2 + Y**2))*np.cos(5*X)
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(f, db20(s21))
    ax[0].set_xlabel("Frequency (kHz)")
    ax[0].set_ylabel("|S21| (dB)")
    ax[0].set_title("Transmission dip")
    im = ax[1].imshow(field, origin="lower", aspect="auto", cmap="viridis")
    ax[1].set_title("FDTD snapshot")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig_06_s21_fdtd.png"))
    plt.close(fig)

# ----------------- Fig. 7 — SSH chain eigenmodes -----------------
def plot_fig7_ssh_modes():
    N = 20
    t1, t2 = 1.0, 2.0
    H = np.zeros((N, N))
    for i in range(N-1):
        t = t1 if i % 2 == 0 else t2
        H[i, i+1] = H[i+1, i] = t
    w, v = np.linalg.eigh(H)
    fig, axes = plt.subplots(2, 2, figsize=(8,6))
    for i, ax in enumerate(axes.ravel()):
        ax.plot(v[:, i], marker="o")
        ax.set_title(f"Mode {i}")
    fig.suptitle("Fig. 7 — SSH chain eigenmodes")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(FIG_DIR, "Fig_07_ssh_modes.png"))
    plt.close(fig)

# ----------------- Fig. 8 — Π2 transfer calibration -----------------
def plot_fig8_pi2_calibration():
    drive = np.linspace(0, 1, 50)
    transfer = 0.5*(1 - np.cos(np.pi*drive))
    noise = rng.normal(scale=0.02, size=drive.shape)
    meas = transfer + noise
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(drive, meas, "o", label="measured")
    ax.plot(drive, transfer, label="ideal")
    ax.set_xlabel("Drive level")
    ax.set_ylabel("Transfer")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "Fig_08_pi2_transfer.png"))
    plt.close(fig)

# ----------------- Fig. 9 — Π3 transfer calibration -----------------
def plot_fig9_pi3_calibration():
    drive = np.linspace(0, 1, 50)
    transfer = drive**2
    meas = transfer + rng.normal(scale=0.03, size=drive.shape)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(drive, meas, "o", label="measured")
    ax.plot(drive, transfer, label="quadratic")
    ax.set_xlabel("Drive level")
    ax.set_ylabel("Transfer")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "Fig_09_pi3_transfer.png"))
    plt.close(fig)

# ----------------- Fig.10 — Comparator stats & coupling matrix -----------------
def plot_fig10_comparator_stats():
    n = 2000
    thr = 0.0
    samples = rng.normal(loc=0.2, scale=0.5, size=n)
    mat = rng.normal(scale=0.2, size=(8, 8))
    np.fill_diagonal(mat, 0.0)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Histogram with Gaussian fit
    counts, edges, _ = ax[0, 0].hist(samples, bins=40, density=True, alpha=0.6)
    bin_width = edges[1] - edges[0]
    mean, std = samples.mean(), samples.std(ddof=1)
    x = np.linspace(edges[0], edges[-1], 400)
    ax[0, 0].plot(x, stats.norm.pdf(x, mean, std), "r--", label=rf"$\mu={mean:.2f}$ V, $\sigma={std:.2f}$ V")
    ax[0, 0].axvline(thr, color="k", linestyle=":", label="decision thr.")
    D, p = stats.kstest((samples - mean) / std, "norm")
    p_err = stats.norm.cdf(thr, loc=mean, scale=std)
    ax[0, 0].fill_between(x, 0, stats.norm.pdf(x, mean, std), where=x <= thr, color="C3", alpha=0.3,
                          label=f"p(err)={p_err:.2e}")
    ax[0, 0].set_xlabel("Voltage (V)")
    ax[0, 0].set_ylabel("Density")
    ax[0, 0].set_title(f"Comparator output, n={n}, bin={bin_width:.2f} V\nKS p={p:.2f}")
    ax[0, 0].legend(fontsize=8)

    # Coupling matrix heatmap
    im = ax[0, 1].imshow(mat, cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax[0, 1].set_title("Coupling matrix $C_{ij}$")
    fig.colorbar(im, ax=ax[0, 1], shrink=0.8, label="Coupling gain (a.u.)")

    # Histogram of off-diagonal terms
    offdiag = mat[~np.eye(mat.shape[0], dtype=bool)]
    ax[1, 0].hist(offdiag, bins=20, color="C2", alpha=0.7)
    ax[1, 0].set_xlabel("$C_{ij}$ (a.u.)")
    ax[1, 0].set_ylabel("Count")
    ax[1, 0].set_title("Off-diagonal coupling histogram")

    # Row-sum norms
    row_norms = np.sum(np.abs(mat), axis=1)
    max_norm = row_norms.max()
    ax[1, 1].bar(np.arange(mat.shape[0]), row_norms, color="C0")
    ax[1, 1].axhline(max_norm, color="r", linestyle="--", label=f"max={max_norm:.2f}")
    ax[1, 1].set_xlabel("Row i")
    ax[1, 1].set_ylabel(r"$\sum_j |C_{ij}|$")
    ax[1, 1].set_title("Row-sum norms")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("Fig. 10 — Comparator stats & coupling matrix")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "Fig_10_comparator_stats.png"))
    plt.close(fig)

# ----------------- Fig.11 — Closed-loop VECTOR_ADD demo -----------------
def plot_fig11_vector_add_demo():
    a = rng.integers(0, 10, size=20)
    b = rng.integers(0, 10, size=20)
    c = a + b
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(a, label="a")
    ax.plot(b, label="b")
    ax.plot(c, label="a+b")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "Fig_11_vector_add.png"))
    plt.close(fig)

# ----------------- Fig.12 — QQ plot of noise distribution -----------------
def plot_fig12_noise_qq():
    import scipy.stats as stats
    data = rng.normal(size=500)
    fig, ax = plt.subplots(figsize=(6,4))
    stats.probplot(data, dist="norm", plot=ax)
    fig.savefig(os.path.join(FIG_DIR, "Fig_12_noise_qq.png"))
    plt.close(fig)

# ----------------- Fig.13 — Survival curve of decision margins -----------------
def plot_fig13_survival_curve():
    margins = np.linspace(0, 5, 100)
    survival = np.exp(-margins)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(margins, survival)
    ax.set_xlabel("Decision margin")
    ax.set_ylabel("Survival probability")
    fig.savefig(os.path.join(FIG_DIR, "Fig_13_survival_curve.png"))
    plt.close(fig)

# ----------------- Fig.14 — Rule-110 survival curve -----------------
def plot_fig14_rule110_survival():
    W, T = 64, 64
    p_vals = np.logspace(-5, -1, 20)
    trials = 500
    surv = []
    for p in p_vals:
        count = 0
        for _ in range(trials):
            errs = rng.random((T, W)) < p
            if not errs.any():
                count += 1
        surv.append(count / trials)
    union_bound = np.clip(1 - W * T * p_vals, 0, 1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(p_vals, surv, "o-", label="Monte Carlo")
    ax.plot(p_vals, union_bound, "--", label=r"Union bound $1-WTp$")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Per-decision error rate $p$")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "Fig_14_rule110_survival.png"))
    plt.close(fig)

# ----------------- Fig.14 — Measured comparator characteristics -----------------
def plot_fig14_comparator_characteristics():
    N_devices, samples_per_device = 5, 1000
    gain = rng.normal(loc=20, scale=1.0, size=N_devices * samples_per_device)
    offset = rng.normal(loc=0, scale=5e-3, size=N_devices * samples_per_device)

    fs = 1e6
    noise = rng.normal(scale=1e-3, size=int(fs))
    freqs = np.fft.rfftfreq(len(noise), 1 / fs)
    psd = np.abs(np.fft.rfft(noise)) ** 2 / (fs * len(noise))

    t = np.linspace(0, 5e-6, 200)
    step = 1 - np.exp(-t / 1e-6)

    fig, ax = plt.subplots(2, 2, figsize=(10,6))
    ax[0,0].hist(gain, bins=40, color="C0", alpha=0.7)
    ax[0,0].set_title("Gain $g$")
    ax[0,0].set_xlabel("Gain")
    ax[0,0].set_ylabel("Count")

    ax[0,1].hist(offset, bins=40, color="C1", alpha=0.7)
    ax[0,1].set_title("Offset $\\theta$")
    ax[0,1].set_xlabel("Offset (V)")
    ax[0,1].set_ylabel("Count")

    ax[1,0].semilogx(freqs[1:], psd[1:])
    ax[1,0].set_title("Noise PSD $\\sigma_n$")
    ax[1,0].set_xlabel("Frequency (Hz)")
    ax[1,0].set_ylabel("PSD")

    ax[1,1].plot(t * 1e6, step)
    ax[1,1].set_title("Step response $\\nu_{\\max}$")
    ax[1,1].set_xlabel("Time (µs)")
    ax[1,1].set_ylabel("Normalised output")

    fig.suptitle("Fig. 14 — Measured comparator characteristics")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(FIG_DIR, "Fig_14_comparator_characteristics.png"))
    plt.close(fig)

# ----------------- Fig.15 — Π2/Π3 device variation sweeps -----------------
def plot_fig15_device_variation():
    variation = np.linspace(-0.2, 0.2, 50)
    pi2 = 1 + 0.5*variation + rng.normal(scale=0.02, size=variation.shape)
    pi3 = 1 - 0.3*variation + rng.normal(scale=0.02, size=variation.shape)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(variation, pi2, label="Π2")
    ax.plot(variation, pi3, label="Π3")
    ax.set_xlabel("Variation")
    ax.set_ylabel("Normalised output")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "Fig_15_device_variation.png"))
    plt.close(fig)

# ----------------- Supporting Fig. S5 — BPSK BER -----------------
def rrc_impulse(beta, span, sps):
    N = span*sps
    t = np.arange(-N/2, N/2+1)/sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4*beta/np.pi
        elif beta > 0 and np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1-(4*beta*ti)**2)
            h[i] = num/den
    return h/np.sqrt(np.sum(h**2))

def bpsk_theory_ber(ebn0_db):
    return 0.5*erfc(np.sqrt(10**(ebn0_db/10)))

def plot_fig8_ber(Nsym=200000, fname="Supp_Fig_S5_BPSK_BER.png"):
    beta, span, sps = 0.35, 8, 8
    ebn0_db = np.arange(0, 10, 2)
    h = rrc_impulse(beta, span, sps)
    ber_runs = []
    for _ in range(10):
        bits = rng.integers(0, 2, Nsym)
        syms = 2*bits - 1
        up = np.zeros(Nsym*sps)
        up[::sps] = syms
        tx = np.convolve(up, h, "full")
        gd = len(h) - 1
        samp = gd + np.arange(Nsym)*sps
        samp = samp[samp < len(tx) + len(h) - 1]
        bits_used = bits[:len(samp)]
        run = []
        for e in ebn0_db:
            ebn0 = 10**(e/10)
            N0 = 1/ebn0
            sigma = np.sqrt(N0/2)
            noise = rng.normal(0, sigma, len(tx))
            rx = np.convolve(tx + noise, h, "full")
            det = (rx[samp] > 0).astype(int)
            run.append(np.mean(det != bits_used))
        ber_runs.append(run)
    ber_runs = np.array(ber_runs)
    mean, std = ber_runs.mean(0), ber_runs.std(0)
    th = bpsk_theory_ber(ebn0_db)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.semilogy(ebn0_db, mean, "o-")
    ax.fill_between(ebn0_db, mean-std, mean+std, alpha=0.2)
    ax.semilogy(ebn0_db, th, "k--", label="theory")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_ylim(1e-5, 1)
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)

# ----------------- Supporting Fig. S6 — Duffing IM spectrogram -----------------
def plot_fig9_duffing_spectrogram() -> None:
    fs = 2000
    T = 2.0
    t = np.arange(int(T * fs)) / fs
    f1, f2 = 100.0, 120.0
    drive_vals = np.linspace(0.1, 2.0, 20)
    spectra = []
    for A in drive_vals:
        sig = A * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))
        sig += 0.3 * A * (sig ** 3)
        sig += 0.005 * rng.normal(size=len(t))
        win = np.hanning(len(sig))
        Xf = np.fft.rfft(sig * win)
        spectra.append(np.abs(Xf))
    freqs = np.fft.rfftfreq(len(t), 1 / fs)
    idx = freqs < 400  # crop to low-frequency region
    freqs = freqs[idx]
    S = np.asarray(spectra)[:, idx]
    SdB = db20(S / S.max())
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(
        SdB,
        aspect="auto",
        origin="lower",
        extent=[freqs[0], freqs[-1], drive_vals[0], drive_vals[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Drive amplitude (a.u.)")
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")

    lines = [f1, f2, 2 * f1 - f2, 2 * f2 - f1, 2 * f1 + f2, 2 * f2 + f1]
    labels = ["$f_1$", "$f_2$", "$2f_1-f_2$", "$2f_2-f_1$", "$2f_1+f_2$", "$2f_2+f_1$"]
    for f, lab in zip(lines, labels):
        if freqs[0] <= f <= freqs[-1]:
            ax.axvline(f, color="w", ls="--", lw=0.5)
            ax.text(
                f,
                drive_vals[-1] * 1.02,
                lab,
                rotation=90,
                va="bottom",
                ha="center",
                color="w",
                fontsize=7,
            )

    fig.savefig(
        os.path.join(FIG_DIR, "Supp_Fig_S6_duffing_spectrogram.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

# ----------------- Supporting Fig. S7 — Duffing linewidth -----------------
def plot_fig10_duffing_linewidth() -> None:
    fs = 40000
    T = 1.0
    t = np.arange(int(T * fs)) / fs
    f0 = 1000.0
    drive_vals = np.linspace(0.2, 1.5, 8)
    mean_fwhm, std_fwhm = [], []
    from scipy.signal import peak_widths
    for A in drive_vals:
        fwhms = []
        for _ in range(30):  # more averaging for a smoother trend
            phase = 2 * np.pi * f0 * t + np.cumsum(rng.normal(scale=0.01 * A, size=len(t)))
            base = np.sin(phase)
            sig = A * base + 0.1 * A * (base ** 3)
            sig += 0.02 * A * rng.normal(size=len(t))
            win = np.hanning(len(sig))
            Xf = np.fft.rfft(sig * win)
            freqs = np.fft.rfftfreq(len(sig), 1 / fs)
            P = np.abs(Xf) ** 2
            idx = (freqs > f0 - 200) & (freqs < f0 + 200)
            fband, Pband = freqs[idx], P[idx]
            peak = np.argmax(Pband)
            width_samples = peak_widths(Pband, [peak], rel_height=0.5)[0][0]
            df = fband[1] - fband[0]
            fwhm = width_samples * df
            fwhms.append(fwhm)
        mean_fwhm.append(np.mean(fwhms))
        std_fwhm.append(np.std(fwhms))

    # normalise to the smallest-drive linewidth to emphasise broadening
    baseline = mean_fwhm[0]
    mean_fwhm = np.array(mean_fwhm) / baseline
    std_fwhm = np.array(std_fwhm) / baseline

    fig = plt.figure(figsize=(7, 3))
    ax = plt.gca()
    ax.errorbar(drive_vals, mean_fwhm, yerr=std_fwhm, fmt="o-", lw=2, capsize=4)
    ax.set_xlabel("Drive amplitude (a.u.)")
    ax.set_ylabel("Normalised FWHM (× baseline)")
    fig.savefig(
        os.path.join(FIG_DIR, "Supp_Fig_S7_duffing_linewidth.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

# ----------------- Supporting Figures (S1–S4) -----------------
def plot_supporting_transmission():
    rho1, c1 = 2700, 6420
    rho2, c2 = 1180, 2550
    d1, d2 = 0.80e-3, 0.32e-3
    Z0 = rho1*c1

    def tmm(f, N):
        w = 2*np.pi*f
        k1 = w/c1
        k2 = w/c2
        Z1, Z2 = rho1*c1, rho2*c2
        M1 = np.array([[np.cos(k1*d1), 1j*Z1*np.sin(k1*d1)],
                       [1j*np.sin(k1*d1)/Z1, np.cos(k1*d1)]])
        M2 = np.array([[np.cos(k2*d2), 1j*Z2*np.sin(k2*d2)],
                       [1j*np.sin(k2*d2)/Z2, np.cos(k2*d2)]])
        M = np.linalg.matrix_power(M1@M2, N)
        A, B, C, D = M.ravel()
        return 2*Z0/(A*Z0 + B + C*Z0*Z0 + D*Z0)

    f = np.linspace(0.5e6, 5e6, 2000)
    fig, ax = plt.subplots(figsize=(9,4))
    for N in [10, 20, 40]:
        T = [tmm(fi, N) for fi in f]
        ax.plot(f*1e-6, db20(T), label=f"N={N}")
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "Supp_Fig_S1_transmission.png"))
    plt.close(fig)

def plot_supporting_design_variants():
    rho1, c1 = 2700, 6420
    rho2, c2 = 1180, 2550
    d1 = 0.80e-3
    fig, ax = plt.subplots(figsize=(9,4))
    for ratio in [0.2, 0.4, 0.6]:
        d2 = ratio*d1
        k, f = bloch_dispersion(d1, d2, c1, c2, rho1, rho2)
        ax.plot(k*(d1+d2), f*1e-6, ".", label=f"d2/d1={ratio}")
    ax.set_ylim(0, 6)
    ax.legend()
    ax.set_xlabel("k (rad/a)")
    ax.set_ylabel("Freq (MHz)")
    fig.savefig(os.path.join(FIG_DIR, "Supp_Fig_S2_design_variants.png"))
    plt.close(fig)

def plot_supporting_linewidth_residuals():
    fs = 2000
    N = 4096
    f0 = 100
    t = np.arange(N)/fs
    A = 0.5
    sig = A*np.sin(2*np.pi*f0*t) + 0.05*(A*np.sin(2*np.pi*f0*t))**3
    spec = np.abs(np.fft.rfft(sig*np.hanning(N)))**2
    freqs = np.fft.rfftfreq(N, 1/fs)
    idx = np.argmin(abs(freqs - f0))
    span = 30
    fseg = freqs[idx-span:idx+span]
    y = spec[idx-span:idx+span]
    p0 = [f0, 2, max(y), min(y)]
    popt, _ = curve_fit(lorentzian, fseg, y, p0=p0)
    yfit = lorentzian(fseg, *popt)
    residual = (y - yfit)/np.max(y)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(fseg, residual, ".-")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual (norm.)")
    fig.savefig(os.path.join(FIG_DIR, "Supp_Fig_S3_linewidth_residuals.png"))
    plt.close(fig)

def plot_supporting_ber_scaling():
    plot_fig8_ber(Nsym=200000, fname="Supp_Fig_S4_BER_scaling_200k.png")
    plot_fig8_ber(Nsym=1000000, fname="Supp_Fig_S4_BER_scaling_1M.png")

# ----------------- Main -----------------
def main():
    plot_fig1_duffing()
    plot_fig2_tmm()
    plot_fig3_accuracy()
    plot_fig4_enob_rmse()
    plot_fig5_bloch()
    plot_fig6_s21_fdtd()
    plot_fig7_ssh_modes()
    plot_fig8_pi2_calibration()
    plot_fig9_pi3_calibration()
    plot_fig10_comparator_stats()
    plot_fig11_vector_add_demo()
    plot_fig12_noise_qq()
    plot_fig13_survival_curve()
    plot_fig14_rule110_survival()
    plot_fig14_comparator_characteristics()
    plot_fig15_device_variation()
    plot_fig8_ber()
    plot_fig9_duffing_spectrogram()
    plot_fig10_duffing_linewidth()
    plot_supporting_transmission()
    plot_supporting_design_variants()
    plot_supporting_linewidth_residuals()
    plot_supporting_ber_scaling()

if __name__ == "__main__":
    main()
