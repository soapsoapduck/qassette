import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qassette.effects import build_fx_circuit, parse_fx
from qassette.encode import encode_frame
from qassette.formats import load_qbits, save_qbits
from qassette.memory import MemoryParams, TapeEvaluator, memory_distribution, sample_qubit_phase
from qassette.readout import homodyne_statevector

N = 256


def _frame_circuit(spec=""):
    rng = np.random.default_rng(42)
    frame = rng.normal(size=N)
    qc, norm = encode_frame(frame, fx=build_fx_circuit(8, parse_fx(spec, 8)))
    return qc, frame / norm


def _zero_params(**kw):
    return MemoryParams(t=0.0, **kw)


@pytest.mark.parametrize("spec", ["", "lowpass:30"])
@pytest.mark.parametrize("mode", ["z", "homodyne"])
def test_zero_storage_matches_qiskit_circuits(spec, mode):
    """t=0 memory path must reproduce the circuit distributions exactly."""
    qc, _ = _frame_circuit(spec)
    phi = Statevector(qc).data
    p = memory_distribution(phi, 8, _zero_params(trajectories=2), np.random.default_rng(0), mode)
    if mode == "z":
        ref = np.abs(phi) ** 2
    else:
        ref = np.abs(homodyne_statevector(qc, data_qubits=8).data) ** 2
    assert np.max(np.abs(p - ref)) < 1e-12


def test_fid_coherence_hits_t2_star():
    p = MemoryParams(t=64e-6)  # t = T2*
    rng = np.random.default_rng(7)
    phases = np.array([sample_qubit_phase(p, rng) for _ in range(20_000)])
    coh = abs(np.mean(np.exp(1j * phases)))
    assert 0.25 < coh < 0.42  # ~1/e, slightly below due to the telegraph part


def test_echo_ordering_at_400us():
    rng = np.random.default_rng(11)
    coh = {}
    for dd in ("none", "hahn", "cpmg"):
        p = MemoryParams(t=400e-6, dd=dd)
        ph = np.array([sample_qubit_phase(p, rng) for _ in range(5000)])
        coh[dd] = abs(np.mean(np.exp(1j * ph)))
    assert coh["none"] < coh["hahn"] <= coh["cpmg"] + 0.03
    assert coh["hahn"] > 2 * coh["none"]


def test_signal_survives_short_storage_dies_at_long():
    qc, psi = _frame_circuit()
    phi = Statevector(qc).data
    ests = {}
    for t in (10e-6, 2e-3):
        p = memory_distribution(
            phi, 8, MemoryParams(t=t, trajectories=32), np.random.default_rng(3), "homodyne"
        )
        ests[t] = np.sqrt(N) * (p[:N] - p[N:])
    # indices carry up to 8 qubits' worth of phase noise, so even 10us
    # (t = T2*/6.4) costs a bit of amplitude; it must still dwarf 2 ms
    assert np.linalg.norm(ests[10e-6]) > 0.75
    assert np.linalg.norm(ests[2e-3]) < 0.3  # far beyond T2*: coherence gone
    assert np.corrcoef(ests[10e-6], psi)[0, 1] > 0.98


def test_dd_preserves_more_signal():
    qc, _ = _frame_circuit()
    phi = Statevector(qc).data
    norms = {}
    for dd in ("none", "hahn"):
        p = memory_distribution(
            phi, 8, MemoryParams(t=300e-6, dd=dd, trajectories=48), np.random.default_rng(5), "homodyne"
        )
        norms[dd] = float(np.linalg.norm(np.sqrt(N) * (p[:N] - p[N:])))
    assert norms["hahn"] > norms["none"] + 0.1


@pytest.mark.parametrize("spec", ["", "ringmod:4", "lowpass:30"])
def test_tape_evaluator_matches_statevector(spec):
    ev = TapeEvaluator()
    for seed in (1, 2):  # two frames reuse the cached operator
        frame = np.random.default_rng(seed).normal(size=N)
        qc, _ = encode_frame(frame, fx=build_fx_circuit(8, parse_fx(spec, 8)))
        fast = ev.output_vector(qc)
        exact = Statevector(qc).data
        assert np.max(np.abs(fast - exact)) < 1e-10


def test_memory_recorded_in_header(tmp_path):
    mem = MemoryParams(t=300e-6, dd="hahn", trajectories=4)
    outs = np.zeros((2, 8), dtype=np.uint16)
    path = tmp_path / "m.qbits"
    save_qbits(path, outs, sr=16000, N=256, hop=128, length=1, mode="z", shots=8,
               norms=[1.0, 1.0], memory=mem.header_dict())
    h = load_qbits(path).header
    assert h["memory"]["dd"] == "hahn"
    assert h["memory"]["t"] == pytest.approx(300e-6)
    assert h["memory"]["T1"] == pytest.approx(25e-3)
