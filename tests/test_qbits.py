import json
from pathlib import Path

import numpy as np
import pytest

from qassette.audio_io import load_audio
from qassette.decode import decode_qbits, frames_to_audio
from qassette.encode import encode_frame, encode_signal
from qassette.formats import load_qbits, save_qbits, to_ascii
from qassette.readout import (
    estimate_homodyne,
    estimate_z,
    homodyne_statevector,
    read_exact,
    sample_statevector,
    sample_z,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SINE = SAMPLES_DIR / "sine_440hz_2s.wav"
SHOT_LADDER = (256, 4096, 65536)


@pytest.fixture(scope="module")
def sine_tapes(tmp_path_factory):
    """Homodyne-sample the sine at several shot counts; one sv evolve per frame."""
    tmp = tmp_path_factory.mktemp("qbits")
    x = load_audio(SINE)
    encoded = encode_signal(x, N=256, hop=128)
    norms = [n for _, n in encoded]
    svs = [homodyne_statevector(qc) for qc, _ in encoded]
    tapes = {}
    for shots in SHOT_LADDER:
        outcomes = np.stack(
            [sample_statevector(sv, shots, seed=shots + i) for i, sv in enumerate(svs)]
        )
        path = tmp / f"sine_{shots}.qbits"
        save_qbits(
            path, outcomes, sr=16000, N=256, hop=128, length=len(x),
            mode="homodyne", shots=shots, norms=norms,
        )
        tapes[shots] = path
    return x, tapes


def test_homodyne_identity_exact():
    """Exact probabilities (no sampling) must satisfy sqrt(N)(P0-P1) = psi."""
    rng = np.random.default_rng(7)
    frame = rng.normal(size=8)
    qc, norm = encode_frame(frame)
    psi = frame / norm
    probs = np.abs(homodyne_statevector(qc).data) ** 2
    est = np.sqrt(8) * (probs[:8] - probs[8:])
    assert np.max(np.abs(est - psi)) < 1e-10


def test_z_magnitudes_statevector_and_aer():
    frame = np.zeros(8)
    frame[1], frame[6] = 0.6, -0.8
    qc, _ = encode_frame(frame)
    expected = np.abs(read_exact(qc))
    est = estimate_z(sample_z(qc, 50_000, seed=11), 8)
    assert np.max(np.abs(est - expected)) < 0.02
    est_aer = estimate_z(sample_z(qc, 8192, use_aer=True, seed=11), 8)
    assert np.max(np.abs(est_aer - expected)) < 0.05
    assert np.all(est >= 0)  # magnitudes only: signs are lost


def test_homodyne_decode_correlation(sine_tapes):
    x, tapes = sine_tapes
    rec = decode_qbits(tapes[65536])
    corr = np.corrcoef(rec, x)[0, 1]
    print(f"\nhomodyne 65536-shot decode correlation: {corr:.5f}")
    assert corr > 0.99


def test_error_scales_with_shots(sine_tapes):
    x, tapes = sine_tapes
    rms = {s: float(np.sqrt(np.mean((decode_qbits(p) - x) ** 2))) for s, p in tapes.items()}
    print("\nshots    RMS error")
    for s in SHOT_LADDER:
        print(f"{s:>6}   {rms[s]:.5f}")
    slope = np.polyfit(np.log(SHOT_LADDER), np.log([rms[s] for s in SHOT_LADDER]), 1)[0]
    print(f"log-log slope: {slope:.3f} (expect ~ -0.5)")
    assert rms[256] > rms[4096] > rms[65536]
    assert -0.7 < slope < -0.3


def test_qbits_round_trip_and_ascii(sine_tapes):
    _, tapes = sine_tapes
    qb = load_qbits(tapes[4096])
    assert qb.header["mode"] == "homodyne"
    assert qb.header["shots"] == 4096
    assert qb.outcomes.dtype == np.uint16
    art = to_ascii(qb, frame=10, max_shots=16)
    print("\n" + art)
    shot_lines = art.splitlines()[2:]
    assert len(shot_lines) == 16
    for line in shot_lines:
        anc, _, data = line.partition(" | ")
        assert len(anc) == 1 and set(anc + data) <= {"0", "1"}
        assert len(data) == 8  # total bitstring width n+1 = 9


def test_load_qbits_rejects_corrupt_header(tmp_path, sine_tapes):
    _, tapes = sine_tapes
    qb = load_qbits(tapes[256])
    bad = dict(qb.header)
    bad["N"] = 128  # wrong: outcomes carry 9-bit homodyne values
    path = tmp_path / "corrupt.qbits"
    with open(path, "wb") as f:
        np.savez(f, header=json.dumps(bad), outcomes=qb.outcomes)
    with pytest.raises(ValueError):
        load_qbits(path)


def test_decode_matches_direct_estimation(sine_tapes):
    """decode_qbits(path) must equal estimating from the in-memory outcomes."""
    x, tapes = sine_tapes
    qb = load_qbits(tapes[4096])
    amps = [estimate_homodyne(row, 256) for row in qb.outcomes]
    direct = frames_to_audio(amps, qb.header["norms"], hop=128, length=len(x))
    assert np.array_equal(decode_qbits(tapes[4096]), direct)
