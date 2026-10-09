"""Decode amplitude vectors back into audio."""

import numpy as np

from .audio_io import overlap_add


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
