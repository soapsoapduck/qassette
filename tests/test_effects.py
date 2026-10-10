import numpy as np
import pytest
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

from qassette.effects import build_fx_circuit
from qassette.encode import encode_frame, encode_signal
from qassette.readout import estimate_z, read_exact, sample_z


def _random_psi(n_samples, seed=3):
    rng = np.random.default_rng(seed)
    frame = rng.normal(size=n_samples)
    qc, norm = encode_frame(frame)
    return qc, frame / norm


def test_qft_iqft_identity():
    qc, psi = _random_psi(256)
    qc.compose(build_fx_circuit(8, ()), inplace=True)
    assert np.max(np.abs(read_exact(qc) - psi)) < 1e-10


def test_qft_matches_numpy_fft():
    """QFTGate output index k = conj(np.fft.fft(psi))[k] / sqrt(N); for real
    psi that is np.fft.fft(psi)[(N - k) % N] / sqrt(N) (negated freq axis)."""
    qc, psi = _random_psi(256)
    qc.append(QFTGate(8), range(8))
    out = Statevector(qc).data
    expected = np.conj(np.fft.fft(psi)) / np.sqrt(256)
    assert np.max(np.abs(out - expected)) < 1e-10
    reordered = (np.fft.fft(psi) / np.sqrt(256))[(-np.arange(256)) % 256]
    assert np.max(np.abs(out - reordered)) < 1e-10


def test_encode_signal_includes_fx_slot():
    x = np.sin(2 * np.pi * 440 * np.arange(800) / 16000).astype(np.float32)
    encoded = encode_signal(x, N=256, hop=128)
    assert all(any(inst.operation.name == "qft" for inst in qc.data) for qc, _ in encoded)
    # empty chain is still the identity end to end
    from qassette.decode import frames_to_audio

    amps = [read_exact(qc) for qc, _ in encoded]
    rec = frames_to_audio(amps, [n for _, n in encoded], hop=128, length=len(x))
    assert np.max(np.abs(rec - x)) < 1e-6


def test_freq_basis_outcomes_are_frequency_bins():
    """A 440 Hz frame measured in basis='freq' concentrates on folded bin
    round(440/16000*256) = 7 (indices 7 and 249 before folding)."""
    t = np.arange(256) / 16000
    frame = np.sin(2 * np.pi * 440 * t)
    qc, _ = encode_frame(frame)
    outcomes = sample_z(qc, 4096, seed=42, basis="freq")
    mags = estimate_z(outcomes, 256)
    folded = mags[: 129] ** 2
    folded[1:128] += mags[255:128:-1] ** 2
    assert int(np.argmax(folded)) == 7


def test_sample_z_rejects_bad_basis():
    qc, _ = _random_psi(8)
    with pytest.raises(ValueError, match="basis"):
        sample_z(qc, 16, basis="wavelet")
