import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qassette.effects import build_fx_circuit, chain_names, fx_chirp, parse_fx
from qassette.encode import encode_frame
from qassette.readout import homodyne_statevector

N = 256


def _signed(m):
    return ((m + N // 2) % N) - N // 2


def _chirp_ref(x, c):
    """Numpy reference: Y_m = X_m * exp(+1j*pi*c*ms*|ms|/N), Nyquist untouched."""
    ms = _signed(np.arange(N))
    phase = np.exp(1j * np.pi * c * ms * np.abs(ms) / N)
    phase[N // 2] = 1.0
    return np.fft.ifft(np.fft.fft(x) * phase)


def _apply(frame, spec):
    chain = parse_fx(spec, 8)
    qc, norm = encode_frame(frame, fx=build_fx_circuit(8, chain))
    return Statevector(qc).data, frame / norm


@pytest.fixture()
def frame():
    rng = np.random.default_rng(55)
    return rng.normal(size=N)


def test_delay_matches_roll(frame):
    out, psi = _apply(frame, "delay:3")
    ref = np.roll(psi, 3)
    assert np.max(np.abs(out - ref)) < 1e-9
    assert np.max(np.abs(out.imag)) < 1e-9  # real stays real


def test_chirp_matches_fft_reference(frame):
    out, psi = _apply(frame, "chirp:0.5")
    ref = _chirp_ref(psi, 0.5)
    assert np.max(np.abs(out - ref)) < 1e-9
    assert np.max(np.abs(out.imag)) < 1e-9  # conjugate-symmetric phase keeps it real


def test_ringmod_matches_modulation(frame):
    out, psi = _apply(frame, "ringmod:4")
    n_idx = np.arange(N)
    ref = psi * np.exp(-2j * np.pi * 4 * n_idx / N)
    assert np.max(np.abs(out - ref)) < 1e-9
    # invisible to Z readout: per-sample magnitudes unchanged
    assert np.max(np.abs(np.abs(out) - np.abs(psi))) < 1e-9


def test_ringmod_homodyne_reads_cosine(frame):
    chain = parse_fx("ringmod:4", 8)
    qc, norm = encode_frame(frame, fx=build_fx_circuit(8, chain))
    probs = np.abs(homodyne_statevector(qc).data) ** 2
    est = np.sqrt(N) * (probs[:N] - probs[N:])
    expected = (frame / norm) * np.cos(2 * np.pi * 4 * np.arange(N) / N)
    assert np.max(np.abs(est - expected)) < 1e-9


def test_chain_equals_sequential(frame):
    out, psi = _apply(frame, "delay:3,chirp:0.5")
    ref = _chirp_ref(np.roll(psi, 3), 0.5)
    assert np.max(np.abs(out - ref)) < 1e-9


def test_parse_fx():
    chain = parse_fx("delay:3,chirp:0.5,ringmod:4", 8)
    assert chain_names(chain) == ["delay:3", "chirp:0.5", "ringmod:4"]
    assert parse_fx("", 8) == ()
    with pytest.raises(ValueError, match="unknown effect"):
        parse_fx("reverb:9", 8)
    with pytest.raises(ValueError, match="bad argument"):
        parse_fx("delay:abc", 8)


def test_chirp_nyquist_untouched():
    diag = fx_chirp(8, 0.7).data[0].operation.params
    assert diag[N // 2] == 1.0
