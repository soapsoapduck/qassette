"""Visualization: spectrograms built from raw quantum shot histograms."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from .formats import Qbits

_EPS = 1e-12


def quantum_spectrogram(qbits: Qbits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrogram (freqs, times, power) from frequency-basis shot histograms.

    Requires mode="z", basis="freq". Per frame, bin probabilities
    count_k/shots are scaled by the stored norm^2 to absolute power, then
    negative frequencies are folded: Qiskit's QFT negates the frequency
    axis relative to numpy (see effects.py), but |X_k| pairs k with N-k
    either way, so the folded spectrum is convention-independent.
    """
    h = qbits.header
    if h["mode"] != "z" or h.get("basis") != "freq":
        raise ValueError(
            f"quantum_spectrogram needs mode='z', basis='freq' outcomes; "
            f"got mode={h['mode']!r}, basis={h.get('basis')!r}"
        )
    N, sr, hop, shots = h["N"], h["sr"], h["hop"], h["shots"]
    num_frames = qbits.outcomes.shape[0]
    power = np.zeros((N // 2 + 1, num_frames))
    for i, (row, norm) in enumerate(zip(qbits.outcomes, h["norms"])):
        p = np.bincount(row, minlength=N) / shots * norm**2
        folded = np.empty(N // 2 + 1)
        folded[0] = p[0]
        folded[N // 2] = p[N // 2]
        folded[1 : N // 2] = p[1 : N // 2] + p[N - 1 : N // 2 : -1]
        power[:, i] = folded
    freqs = np.arange(N // 2 + 1) * sr / N
    times = np.arange(num_frames) * hop / sr
    return freqs, times, power


def spectrogram_comparison(qbits: Qbits, x: np.ndarray, sr: int) -> plt.Figure:
    """Side-by-side figure: quantum shot-histogram spectrogram vs scipy's."""
    h = qbits.header
    freqs, times, power = quantum_spectrogram(qbits)
    f2, t2, s2 = sps.spectrogram(
        x, fs=sr, window="hann", nperseg=h["N"], noverlap=h["N"] - h["hop"], mode="psd"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (f, t, s, title) in zip(
        (ax1, ax2),
        (
            (freqs, times, power, f"quantum ({h['shots']} shots/frame)"),
            (f2, t2, s2, "classical (scipy)"),
        ),
    ):
        db = 10 * np.log10(s + _EPS)
        mesh = ax.pcolormesh(t, f, db, shading="auto", vmin=db.max() - 70, vmax=db.max())
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        fig.colorbar(mesh, ax=ax, label="dB")
    ax1.set_ylabel("frequency [Hz]")
    fig.tight_layout()
    return fig
