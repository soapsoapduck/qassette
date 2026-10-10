"""Decode amplitude vectors (or .qbits tapes) back into audio."""

from pathlib import Path

import numpy as np

from .audio_io import overlap_add
from .formats import load_qbits
from .readout import estimate_homodyne, estimate_z


def frames_to_audio(
    amplitudes_list: list[np.ndarray],
    norms: list[float],
    hop: int,
    length: int,
) -> np.ndarray:
    """Rescale unit amplitude vectors by their norms and overlap-add.

    Inverse of encode_signal + read_exact: each amplitude vector times its
    norm is a windowed frame in the frame_signal convention.
    """
    frames = np.stack(
        [np.asarray(a, dtype=np.float64) * n for a, n in zip(amplitudes_list, norms, strict=True)]
    )
    return overlap_add(frames, hop=hop, length=length)


def decode_qbits(path: str | Path, renormalize: bool = False) -> np.ndarray:
    """Reconstruct a waveform using ONLY a .qbits file.

    Estimates each frame's amplitude vector from its stored shot outcomes
    (signed for homodyne, magnitudes only for z), rescales by the stored
    norms, and overlap-adds. Sample rate is available in the file header
    via formats.load_qbits.

    renormalize=True scales any frame estimate whose L2 norm exceeds 1 back
    to unit norm (shot noise can push estimates outside the physical ball,
    which clips loud at low shot counts). Default off: the raw estimator is
    unbiased, so leave this disabled for analysis.
    """
    qb = load_qbits(path)
    h = qb.header
    N = h["N"]
    estimate = estimate_homodyne if h["mode"] == "homodyne" else estimate_z
    amps = []
    for row in qb.outcomes:
        a = estimate(row, N)
        if renormalize:
            nrm = np.linalg.norm(a)
            if nrm > 1.0:
                a = a / nrm
        amps.append(a)
    return frames_to_audio(amps, h["norms"], hop=h["hop"], length=h["length"])
