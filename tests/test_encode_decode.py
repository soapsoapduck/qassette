import time
from pathlib import Path

import numpy as np
import pytest

from qassette.audio_io import load_audio
from qassette.decode import frames_to_audio
from qassette.encode import encode_frame, encode_signal
from qassette.readout import read_exact

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@pytest.mark.parametrize("k", [0, 3, 37, 200, 255])
def test_little_endian_index_mapping(k):
    """A frame with its only nonzero sample at index k must populate
    Statevector index k (Qiskit little-endian basis indexing)."""
    frame = np.zeros(256)
    frame[k] = 0.7
    qc, norm = encode_frame(frame)
    amps = read_exact(qc)
    assert norm == pytest.approx(0.7)
    assert np.argmax(np.abs(amps)) == k
    assert amps[k] == pytest.approx(1.0)


def test_zero_frame_flagged():
    qc, norm = encode_frame(np.zeros(256))
    assert norm == 0.0
    amps = read_exact(qc)
    assert amps[0] == pytest.approx(1.0)  # |0...0>
    assert np.allclose(amps[1:], 0.0)


@pytest.mark.parametrize(
    "wav", sorted(SAMPLES_DIR.glob("*.wav")), ids=lambda p: p.stem
)
def test_end_to_end(wav):
    x = load_audio(wav)
    t0 = time.perf_counter()
    encoded = encode_signal(x, N=256, hop=128)
    t1 = time.perf_counter()
    amps = [read_exact(qc) for qc, _ in encoded]
    t2 = time.perf_counter()
    rec = frames_to_audio(amps, [n for _, n in encoded], hop=128, length=len(x))
    t3 = time.perf_counter()
    print(
        f"\n{wav.stem}: {len(encoded)} frames | encode {t1 - t0:.2f}s | "
        f"readout {t2 - t1:.2f}s | decode {t3 - t2:.3f}s"
    )
    assert np.max(np.abs(rec - x)) < 1e-6
