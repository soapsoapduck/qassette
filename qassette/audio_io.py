"""Audio I/O and STFT-style framing for qassette.

Framing convention: frame_signal pre-pads the signal with N - hop zeros and
zero-pads the tail, so every real sample has full analysis-window coverage.
overlap_add assumes the same convention and trims the pre-pad back off.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_COLA_TOL = 1e-6


def _periodic_hann(n: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Load an audio file as mono float32 in [-1, 1], resampled to sr."""
    x, file_sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        ratio = Fraction(sr, file_sr)
        x = resample_poly(x, ratio.numerator, ratio.denominator)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def save_audio(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Write a mono signal to disk."""
    sf.write(path, np.asarray(x, dtype=np.float32), sr)


def frame_signal(x: np.ndarray, N: int = 256, hop: int = 128) -> np.ndarray:
    """Slice x into overlapping frames of length N windowed by a periodic Hann.

    The signal is pre-padded with N - hop zeros and the tail is zero-padded so
    that every real sample is covered by a full set of overlapping windows.
    Returns an array of shape (num_frames, N), float32.
    """
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of two")
    if not 0 < hop <= N:
        raise ValueError("hop must be in (0, N]")
    x = np.asarray(x, dtype=np.float64).ravel()
    pad = N - hop
    last = pad + len(x) - 1 if len(x) else 0
    num_frames = last // hop + 1
    padded = np.zeros((num_frames - 1) * hop + N, dtype=np.float64)
    padded[pad : pad + len(x)] = x
    idx = np.arange(N)[None, :] + hop * np.arange(num_frames)[:, None]
    return (padded[idx] * _periodic_hann(N)[None, :]).astype(np.float32)


def overlap_add(frames: np.ndarray, hop: int, length: int) -> np.ndarray:
    """Reconstruct a signal of the given length from frame_signal output.

    Uses analysis-window normalization: the overlapped window sum is verified
    to satisfy COLA (constant overlap-add) over the real-sample region, and
    the accumulated signal is divided by it, so a periodic Hann at 50%
    overlap reconstructs with unity gain.
    """
    frames = np.asarray(frames, dtype=np.float64)
    num_frames, N = frames.shape
    pad = N - hop
    total = (num_frames - 1) * hop + N
    out = np.zeros(total)
    wsum = np.zeros(total)
    window = _periodic_hann(N)
    for k in range(num_frames):
        out[k * hop : k * hop + N] += frames[k]
        wsum[k * hop : k * hop + N] += window
    region = wsum[pad : pad + length]
    if not np.allclose(region, region[0], atol=_COLA_TOL):
        raise ValueError("window/hop combination does not satisfy COLA")
    out[wsum > _COLA_TOL] /= wsum[wsum > _COLA_TOL]
    return out[pad : pad + length].astype(np.float32)
