import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qassette.decode import decode_qbits
from qassette.effects import build_fx_circuit, parse_fx
from qassette.encode import encode_frame, encode_signal
from qassette.formats import load_qbits, save_qbits, to_ascii
from qassette.readout import estimate_z, sample_homodyne, sample_z, success_rate

N = 256
SR = 16000


def _signed(m):
    return ((m + N // 2) % N) - N // 2


def _mask(kind, cutoff, angle=np.pi):
    ks = _signed(np.arange(N))
    reject = np.abs(ks) > cutoff if kind == "lowpass" else np.abs(ks) < cutoff
    m = np.ones(N)
    m[reject] = np.cos(angle / 2)
    return m


def _filtered_state(frame, spec):
    chain = parse_fx(spec, 8)
    qc, norm = encode_frame(frame, fx=build_fx_circuit(8, chain))
    sv = Statevector(qc).data
    assert len(sv) == 2 * N  # data + filter ancilla
    return sv[:N], sv[N:], frame / norm, qc  # kept (f=0), rejected, psi


@pytest.fixture()
def frame():
    return np.random.default_rng(123).normal(size=N)


@pytest.mark.parametrize("kind,cutoff", [("lowpass", 30), ("highpass", 30)])
def test_filter_exact_matches_fft_masking(frame, kind, cutoff):
    kept, _, psi, _ = _filtered_state(frame, f"{kind}:{cutoff}")
    ref = np.fft.ifft(np.fft.fft(psi) * _mask(kind, cutoff))
    assert np.max(np.abs(kept - ref)) < 1e-9  # unrenormalized projection
    rn_kept = kept / np.linalg.norm(kept)
    rn_ref = ref / np.linalg.norm(ref)
    assert np.max(np.abs(rn_kept - rn_ref)) < 1e-9  # renormalized too
    assert np.max(np.abs(kept.imag)) < 1e-9  # even real mask keeps real real


def test_success_rate_is_kept_energy(frame):
    kept, _, psi, _ = _filtered_state(frame, "lowpass:30")
    p_success = float(np.sum(np.abs(kept) ** 2))
    kept_energy = float(np.sum(np.abs(np.fft.fft(psi) * _mask("lowpass", 30)) ** 2) / N)
    assert abs(p_success - kept_energy) < 1e-9


def test_softness_attenuates_instead_of_zeroing(frame):
    kept, _, psi, _ = _filtered_state(frame, "lowpass:30:2.0")
    ref = np.fft.ifft(np.fft.fft(psi) * _mask("lowpass", 30, angle=2.0))
    assert np.max(np.abs(kept - ref)) < 1e-9


def test_fshift_moves_on_bin_tone_by_k_bins():
    tone = np.sin(2 * np.pi * 16 * np.arange(N) / N)  # 1000 Hz, exactly bin 16
    kept, _, psi, qc = _filtered_state(tone, "fshift:8")
    assert np.max(np.abs(kept.imag)) < 1e-9  # SSB shift keeps real real
    spec = np.abs(np.fft.fft(kept))
    folded = spec[: N // 2 + 1].copy()
    folded[1 : N // 2] += spec[N - 1 : N // 2 : -1]
    assert int(np.argmax(folded)) == 24  # 16 + 8 bins = 1500 Hz
    # sampled success tracks the exact projector probability
    exact_p = float(np.sum(np.abs(kept) ** 2))
    outs = sample_z(qc, 8192, seed=5, data_qubits=8)
    kept_frac = float(np.mean((outs >> 8) & 1 == 0))
    assert abs(kept_frac - exact_p) < 0.02
    mags = estimate_z(outs, N, filter_bit=8)
    assert np.max(np.abs(mags - np.abs(kept))) < 0.05


def test_filtered_homodyne_pipeline(tmp_path):
    t = np.arange(3200) / SR
    x = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chain = parse_fx("lowpass:12", 8)
    encoded = encode_signal(x, N=256, hop=128, chain=chain)
    rows = [
        sample_homodyne(qc, 8192, seed=i, data_qubits=8) for i, (qc, _) in enumerate(encoded)
    ]
    rates = [success_rate(r, 8, homodyne=True) for r in rows]
    assert np.mean(rates) > 0.95  # 440 Hz sits inside a bin-12 lowpass
    path = tmp_path / "lp.qbits"
    save_qbits(
        path, np.stack(rows), sr=SR, N=256, hop=128, length=len(x), mode="homodyne",
        shots=8192, norms=[n for _, n in encoded], filter_ancilla=True, success_rates=rates,
    )
    qb = load_qbits(path)
    assert qb.header["filter"] is True
    assert qb.header["success_rates"] == rates
    rec = decode_qbits(path)
    # 440 Hz (bin ~7) passes a lowpass at bin 12 almost untouched
    assert np.corrcoef(rec, x)[0, 1] > 0.9
    art = to_ascii(qb, frame=5, max_shots=4)
    line = art.splitlines()[2]
    anc, _, data = line.partition(" | ")
    assert len(anc.split()) == 2 and len(data) == 8  # "a f | dddddddd"


def test_hard_lowpass_removes_out_of_band_tone(tmp_path):
    t = np.arange(3200) / SR
    x = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chain = parse_fx("lowpass:4", 8)  # cutoff 250 Hz < 440 Hz
    encoded = encode_signal(x, N=256, hop=128, chain=chain)
    rows = [
        sample_homodyne(qc, 4096, seed=i, data_qubits=8) for i, (qc, _) in enumerate(encoded)
    ]
    rates = [success_rate(r, 8, homodyne=True) for r in rows]
    assert np.mean(rates) < 0.2  # nearly all energy is above a bin-4 lowpass
    path = tmp_path / "kill.qbits"
    save_qbits(
        path, np.stack(rows), sr=SR, N=256, hop=128, length=len(x), mode="homodyne",
        shots=4096, norms=[n for _, n in encoded], filter_ancilla=True, success_rates=rates,
    )
    rec = decode_qbits(path)
    assert float(np.mean(rec**2)) < 0.05 * float(np.mean(x**2))


def test_success_rate_removes_homodyne_reference_branch():
    # filter bit 2; 6 of 8 shots have it low -> raw r = 0.75
    outs = np.array([0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b000, 0b011], dtype=np.uint16)
    assert success_rate(outs, 2) == pytest.approx(0.75)
    assert success_rate(outs, 2, homodyne=True) == pytest.approx(0.5)  # p = 2r - 1
    all_high = np.full(8, 0b100, dtype=np.uint16)
    assert success_rate(all_high, 2, homodyne=True) == 0.0  # clamped


def test_parser_rules():
    assert parse_fx("lowpass:40:2.5", 8)[0].name == "lowpass:40:2.5"
    with pytest.raises(ValueError, match="at most one post-selected"):
        parse_fx("lowpass:40,fshift:8", 8)
    with pytest.raises(ValueError, match="fshift amount"):
        parse_fx("fshift:200", 8)
    with pytest.raises(ValueError, match="softness angle"):
        parse_fx("lowpass:40:9", 8)
