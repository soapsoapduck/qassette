from pathlib import Path

import numpy as np
import pytest

from qassette.audio_io import frame_signal, load_audio, overlap_add

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "samples").glob("*.wav"))


@pytest.mark.parametrize("wav", SAMPLES, ids=lambda p: p.stem)
def test_round_trip(wav):
    x = load_audio(wav)
    frames = frame_signal(x, N=256, hop=128)
    rec = overlap_add(frames, hop=128, length=len(x))
    assert rec.shape == x.shape
    assert np.max(np.abs(rec - x)) < 1e-5


def test_samples_exist():
    assert len(SAMPLES) == 3


def test_non_power_of_two_rejected():
    with pytest.raises(AssertionError):
        frame_signal(np.zeros(1000), N=300, hop=150)
