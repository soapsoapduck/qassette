"""Calibrate qassette.memory noise parameters against the paper's
single-qubit coherence times (Bozkurt et al., Nat. Phys. 21, 1469 (2025)):
T2* = 64 us (Ramsey), T2E = 562 us (Hahn), T2CP2 = 1.02 ms (CPMG-2),
T1 = 25 ms.

The optimizer inner loop uses a vectorized dense-grid surrogate of the
same noise process (quasi-static Gaussian + telegraph, identical math) for
speed; the final verification and every published curve are measured with
qassette.memory itself (sample_qubit_phase / noisy_vectors) on one qubit.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qassette.memory import MemoryParams, noisy_vectors  # noqa: E402

TARGETS = {"ramsey": 64e-6, "hahn": 562e-6, "cpmg": 1.02e-3, "t1": 25e-3}
T1_FIXED = 25e-3


# ---------------------------------------------------------------- surrogate
def _surrogate_coherence(ts, dd, sigma, a, gamma, m=3000, steps=240, seed=0):
    """|<exp(i phi)>| on a dense time grid; same statistics as memory.py."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(ts))
    for i, t in enumerate(ts):
        if t == 0:
            out[i] = 1.0
            continue
        dt = t / steps
        grid = (np.arange(steps) + 0.5) * dt
        sign = np.ones(steps)
        if dd == "hahn":
            sign[grid > t / 2] = -1
        elif dd == "cpmg":
            sign[(grid > t / 4) & (grid < 3 * t / 4)] = -1
        qs = sigma * rng.standard_normal((m, 1))
        s = np.where(rng.random((m, 1)) < 0.5, 1.0, -1.0)
        flips = rng.random((m, steps)) < gamma * dt
        s = s * np.cumprod(np.where(flips, -1.0, 1.0), axis=1)
        phi = ((qs + a * s) * sign * dt).sum(axis=1)
        out[i] = np.abs(np.mean(np.exp(1j * phi)))
    return out


def one_over_e_time(ts, cs):
    """First crossing of 1/e, log-linear interpolation."""
    target = np.exp(-1)
    for j in range(1, len(ts)):
        if cs[j] < target <= cs[j - 1]:
            f = (cs[j - 1] - target) / (cs[j - 1] - cs[j])
            return ts[j - 1] + f * (ts[j] - ts[j - 1])
    return ts[-1]


def _echo_times(sigma, a, gamma, m=4000, seed=7):
    ts = np.linspace(10e-6, 3e-3, 36)
    th = one_over_e_time(ts, np.abs(measure_curve("hahn", ts, sigma, a, gamma, m=m, seed=seed)))
    tc = one_over_e_time(ts, np.abs(measure_curve("cpmg", ts, sigma, a, gamma, m=m, seed=seed)))
    return th, tc


def tune(verbose=True):
    """Calibrate directly against qassette.memory (no surrogate).

    Physics constraint (documented in memory.py): with one telegraph
    fluctuator the measured CPMG-2/Hahn ratio tops out around ~1.38 near
    gamma_tls ~ 1/T2E, far from the paper's 1.82 — so the echo times are
    balanced symmetrically around their targets (~+-14%), which is the
    minimax solution and sits inside the +-15% tolerance. sigma_qs is
    then trimmed so the measured (not input) Ramsey time is T2*.
    """
    sigma0 = np.sqrt(2) / TARGETS["ramsey"]
    best = None
    for gamma in (1400.0, 1700.0, 2000.0):
        lo, hi = 2000.0, 40000.0  # bisect a: larger a shortens both echoes
        for _ in range(9):
            a = np.sqrt(lo * hi)
            th, tc = _echo_times(sigma0, a, gamma)
            # balance: want hahn above target by the same factor cpmg is below
            if th / TARGETS["hahn"] > TARGETS["cpmg"] / tc:
                lo = a
            else:
                hi = a
        a = np.sqrt(lo * hi)
        th, tc = _echo_times(sigma0, a, gamma, m=6000, seed=8)
        worst = max(abs(th / TARGETS["hahn"] - 1), abs(tc / TARGETS["cpmg"] - 1))
        if verbose:
            print(f"  gamma={gamma:.0f}: a={a:.0f} hahn {th * 1e6:.0f}us ({th / TARGETS['hahn'] - 1:+.1%}) "
                  f"cpmg {tc * 1e6:.0f}us ({tc / TARGETS['cpmg'] - 1:+.1%})")
        if best is None or worst < best[0]:
            best = (worst, a, gamma)
    _, a, gamma = best
    # trim sigma so the MEASURED Ramsey 1/e time lands on T2*
    ts = np.linspace(1e-6, 300e-6, 40)
    lo_s, hi_s = sigma0 * 0.5, sigma0 * 1.3
    for _ in range(8):
        sigma = np.sqrt(lo_s * hi_s)
        tr = one_over_e_time(ts, np.abs(measure_curve("ramsey", ts, sigma, a, gamma, m=6000, seed=9)))
        if tr > TARGETS["ramsey"]:
            lo_s = sigma
        else:
            hi_s = sigma
    sigma = np.sqrt(lo_s * hi_s)
    if verbose:
        print(f"tuned: sigma_qs={sigma:.4g} rad/s (T2*_input={np.sqrt(2) / sigma * 1e6:.2f}us), "
              f"a_tls={a:.4g} rad/s, gamma_tls={gamma:.4g} /s")
    return sigma, a, gamma


# ------------------------------------------------- memory-based measurement
def _params(t, dd, sigma, a, gamma):
    return MemoryParams(t=t, dd=dd, trajectories=1, T1=T1_FIXED,
                        T2_star=np.sqrt(2) / sigma, gamma_tls=gamma, a_tls=a)


def measure_curve(seq, ts, sigma, a, gamma, m=3000, seed=0):
    """<X> (Ramsey/Hahn/CPMG from |+>) or <1|rho|1> (T1 from |1>),
    measured through qassette.memory.noisy_vectors on one qubit."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(ts))
    for i, t in enumerate(ts):
        if seq == "t1":
            p = _params(t, "none", sigma, a, gamma)
            p.trajectories = m
            vecs = noisy_vectors(np.array([0.0, 1.0]), 1, p, rng)
            out[i] = float(np.mean(np.abs(vecs[:, 1]) ** 2))
        else:
            dd = "none" if seq == "ramsey" else seq
            p = _params(t, dd, sigma, a, gamma)
            p.trajectories = m
            vecs = noisy_vectors(np.array([1.0, 1.0]) / np.sqrt(2), 1, p, rng)
            out[i] = float(np.mean(2 * np.real(np.conj(vecs[:, 0]) * vecs[:, 1])))
    return out


def stretched_exp(t, T, p_exp, A):
    return A * np.exp(-((t / T) ** p_exp))


def fit_curve(ts, cs):
    p0 = [max(one_over_e_time(ts, np.abs(cs)), ts[1]), 1.5, 1.0]
    try:
        popt, _ = curve_fit(stretched_exp, ts, cs, p0=p0, maxfev=20000)
        return popt
    except RuntimeError:
        return p0


def verify(sigma, a, gamma, m=6000, seed=99):
    """Memory-measured 1/e times for all four sequences."""
    grids = {
        "ramsey": np.linspace(1e-6, 300e-6, 30),
        "hahn": np.linspace(10e-6, 3e-3, 30),
        "cpmg": np.linspace(10e-6, 3e-3, 30),
        "t1": np.linspace(1e-4, 100e-3, 30),
    }
    times = {}
    for seq, ts in grids.items():
        cs = measure_curve(seq, ts, sigma, a, gamma, m=m, seed=seed)
        times[seq] = one_over_e_time(ts, np.abs(cs))
    return times


SEQ_META = {
    "ramsey": ("Ramsey (free induction)", r"$T_2^*$", 64e-6, 1e6, "µs", "#2a78d6"),
    "hahn": ("Hahn echo", r"$T_{2E}$", 562e-6, 1e6, "µs", "#eb6834"),
    "cpmg": ("CPMG-2", r"$T_{2,CP2}$", 1.02e-3, 1e6, "µs", "#1baf7a"),
    "t1": ("Energy relaxation", r"$T_1$", 25e-3, 1e3, "ms", "#eda100"),
}

_INK, _INK2, _MUTED = "#0b0b0b", "#52514e", "#898781"
_GRID, _BASE, _SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def make_figure(sigma, a, gamma, out_path, m=6000, seed=99):
    """Four coherence decays, measured through qassette.memory, with
    stretched-exponential fits and paper-value annotations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = {
        "ramsey": np.linspace(1e-6, 300e-6, 36),
        "hahn": np.linspace(10e-6, 3e-3, 36),
        "cpmg": np.linspace(10e-6, 3e-3, 36),
        "t1": np.linspace(1e-4, 100e-3, 36),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=_SURF)
    for ax, (seq, ts) in zip(axes.ravel(), grids.items()):
        title, sym, paper, scale, unit, color = SEQ_META[seq]
        cs = measure_curve(seq, ts, sigma, a, gamma, m=m, seed=seed)
        T, p_exp, A = fit_curve(ts, np.abs(cs))
        tfine = np.linspace(ts[0], ts[-1], 400)
        ax.set_facecolor(_SURF)
        ax.plot(tfine * scale, stretched_exp(tfine, T, p_exp, A), color=color, lw=2, zorder=2)
        ax.plot(ts * scale, cs, "o", ms=4.5, color=color, mec="white", mew=0.7, zorder=3)
        ax.axhline(np.exp(-1), color=_GRID, lw=1, zorder=1)
        sim_1e = one_over_e_time(ts, np.abs(cs))
        ax.annotate(
            f"simulated {sym} = {sim_1e * scale:.0f} {unit}\n"
            f"paper: {paper * scale:.0f} {unit}  ({sim_1e / paper - 1:+.0%})",
            xy=(0.97, 0.78), xycoords="axes fraction", ha="right",
            fontsize=9.5, color=_INK2, linespacing=1.5,
        )
        ax.set_title(title, fontsize=11.5, color=_INK, loc="left", pad=8)
        ax.set_xlabel(f"storage time [{unit}]", fontsize=9.5, color=_MUTED)
        ax.set_ylabel(r"$\langle X \rangle$" if seq != "t1" else r"$P(|1\rangle)$",
                      fontsize=9.5, color=_MUTED)
        ax.tick_params(colors=_MUTED, labelsize=8.5)
        ax.grid(axis="y", color=_GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_BASE)
        ax.set_ylim(min(-0.08, float(np.min(cs)) - 0.05), 1.05)
    fig.suptitle(
        "qassette.memory vs Bozkurt et al., Nat. Phys. 21, 1469 (2025) — "
        "single-qubit coherence, simulated",
        fontsize=12.5, color=_INK, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURF)
    return fig


if __name__ == "__main__":
    sigma, a, gamma = tune()
    times = verify(sigma, a, gamma)
    print(f"{'seq':<8} {'simulated':>12} {'target':>10} {'dev':>8}")
    for seq, tgt in TARGETS.items():
        dev = times[seq] / tgt - 1
        flag = "ok" if abs(dev) <= 0.15 else "OUT"
        print(f"{seq:<8} {times[seq] * 1e6:>10.1f}us {tgt * 1e6:>8.0f}us {dev:>+7.1%} {flag}")
