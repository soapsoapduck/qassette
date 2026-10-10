import numpy as np
import pytest

from qassette.encode import encode_signal
from qassette.formats import Qbits
from qassette.readout import sample_z
from qassette.viz import quantum_spectrogram, spectrogram_comparison


@pytest.fixture(scope="module")
def freq_qbits():
    sr = 16000
    t = np.arange(2400) / sr
    x = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    encoded = encode_signal(x, N=256, hop=128)
    rows = [sample_z(qc, 2048, seed=i, basis="freq") for i, (qc, _) in enumerate(encoded)]
    header = {
        "version": 1, "sr": sr, "N": 256, "hop": 128, "length": len(x),
        "mode": "z", "basis": "freq", "shots": 2048,
        "norms": [n for _, n in encoded], "effects": [],
    }
    return Qbits(header=header, outcomes=np.stack(rows)), x, sr


def test_quantum_spectrogram_shapes_and_peak(freq_qbits):
    qb, x, sr = freq_qbits
    freqs, times, power = quantum_spectrogram(qb)
    assert power.shape == (129, qb.outcomes.shape[0])
    assert freqs[0] == 0 and freqs[-1] == sr / 2
    mid = power[:, power.shape[1] // 2]
    assert abs(freqs[int(np.argmax(mid))] - 440) < sr / 256  # peak within one bin


def test_quantum_spectrogram_rejects_time_basis(freq_qbits):
    qb, _, _ = freq_qbits
    bad = Qbits(header={**qb.header, "basis": "time"}, outcomes=qb.outcomes)
    with pytest.raises(ValueError, match="freq"):
        quantum_spectrogram(bad)


def test_spectrogram_comparison_saves(tmp_path, freq_qbits):
    qb, x, sr = freq_qbits
    fig = spectrogram_comparison(qb, x, sr)
    out = tmp_path / "spec.png"
    fig.savefig(out)
    assert out.stat().st_size > 10_000
