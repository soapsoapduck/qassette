"""Simulated mechanical quantum memory, modeled on Bozkurt et al.,
Nature Physics 21, 1469 (2025): each data qubit is swapped into its own
phononic oscillator after the inverse QFT, stored for time t, and swapped
back for readout. Readout ancillas (homodyne/filter) are never stored.

Noise model (per data qubit, per stored frame):
- Energy decay: amplitude damping, gamma = 1 - exp(-t/T1), unraveled as
  quantum-jump trajectories (expectation equals the exact channel).
- Dephasing: a per-qubit frequency offset delta(t) = quasi-static Gaussian
  (sigma = sqrt(2)/T2*, so free induction decays to 1/e at T2*) plus a
  random telegraph component (+-a_tls, switching rate gamma_tls),
  integrated exactly between dynamical-decoupling pulse boundaries and
  applied as RZ(integral delta dt). Deliberately NOT thermal_relaxation /
  Aer dephasing: those are memoryless, and an echo cannot refocus
  memoryless noise — the whole point of hahn/cpmg here.
- Decoupling: dd in {none, hahn, cpmg}: X pulses at t/2 (hahn) or t/4 and
  3t/4 (cpmg), modeled as sign flips of the accumulated phase per segment
  (pulse parity is compensated, i.e. pulses are treated as perfect).

Calibration note: a SINGLE telegraph fluctuator cannot reproduce all
three paper coherence times at once. The defaults (gamma_tls = 1780 /s,
a_tls = 6494 rad/s) reproduce T2* = 64 us and T2_hahn ~ 571 us vs the
562 us target, but give CPMG-2 ~ 790 us against the 1.02 ms target —
the paper's noise is an ensemble of TLS, not one. Targets are recorded
in the dataclass for reference.

Trajectory averaging: each frame draws `trajectories` independent noise
realizations; shots are sampled from the uniform mixture of the
trajectory readout distributions, which are computed analytically
(identical math to the homodyne/z circuits — verified in tests at t=0
against the qiskit statevector path to machine precision).
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector

_DD = ("none", "hahn", "cpmg")


class TapeEvaluator:
    """Fast tape-circuit output vectors.

    Every circuit on a tape is StatePreparation(psi) followed by one
    shared fx block, so the block's Operator matrix is built once (keyed
    by a hash of its instructions) and each frame costs a single
    matrix-vector product instead of a full qiskit statevector evolution.
    Falls back transparently when the trailing block changes.
    """

    def __init__(self):
        self._key = None
        self._op = None

    @staticmethod
    def _fingerprint(instructions) -> str:
        h = hashlib.sha1()
        for inst in instructions:
            h.update(inst.operation.name.encode())
            for p in inst.operation.params:
                h.update(np.asarray(p, dtype=complex).tobytes())
        return h.hexdigest()

    def output_vector(self, circuit: QuantumCircuit) -> np.ndarray:
        data = list(circuit.data)
        has_prep = bool(data) and isinstance(data[0].operation, StatePreparation)
        rest = data[1:] if has_prep else data
        dim = 1 << circuit.num_qubits
        key = (circuit.num_qubits, self._fingerprint(rest))
        if key != self._key:
            if rest:
                block = QuantumCircuit(circuit.num_qubits)
                for inst in rest:
                    block.append(
                        inst.operation, [circuit.find_bit(q).index for q in inst.qubits]
                    )
                self._op = Operator(block).data
            else:
                self._op = np.eye(dim, dtype=complex)
            self._key = key
        psi = np.zeros(dim, dtype=complex)
        if has_prep:
            amps = np.asarray(data[0].operation.params, dtype=complex)
            psi[: len(amps)] = amps  # prep acts on the low data qubits
        else:
            psi[0] = 1.0
        return self._op @ psi


@dataclass
class MemoryParams:
    t: float  # storage time [s]
    dd: str = "none"
    trajectories: int = 8
    T1: float = 25e-3
    T2_star: float = 64e-6
    T2_hahn_target: float = 562e-6
    T2_cpmg2_target: float = 1.02e-3
    gamma_tls: float = 1780.0
    a_tls: float = 6494.0
    transfer_infidelity: float = 0.0  # per swap-in/out, per qubit

    def __post_init__(self):
        if self.dd not in _DD:
            raise ValueError(f"dd must be one of {_DD}, got {self.dd!r}")
        if self.t < 0 or self.trajectories < 1:
            raise ValueError("storage time must be >= 0 and trajectories >= 1")

    @property
    def sigma_qs(self) -> float:
        return float(np.sqrt(2.0) / self.T2_star)

    def header_dict(self) -> dict:
        return {
            "t": self.t, "dd": self.dd, "trajectories": self.trajectories,
            "T1": self.T1, "T2_star": self.T2_star,
            "gamma_tls": self.gamma_tls, "a_tls": self.a_tls,
            "transfer_infidelity": self.transfer_infidelity,
        }


def _dd_boundaries(t: float, dd: str) -> list[tuple[float, float, int]]:
    """(start, end, sign) segments; sign flips at each X pulse."""
    if dd == "hahn":
        return [(0.0, t / 2, 1), (t / 2, t, -1)]
    if dd == "cpmg":
        return [(0.0, t / 4, 1), (t / 4, 3 * t / 4, -1), (3 * t / 4, t, 1)]
    return [(0.0, t, 1)]


def sample_qubit_phase(p: MemoryParams, rng: np.random.Generator) -> float:
    """One realization of integral(sign(t') * delta(t') dt') over storage."""
    segs = _dd_boundaries(p.t, p.dd)
    signed_time = sum(s * (b - a) for a, b, s in segs)  # 0 under any echo
    phi = p.sigma_qs * rng.standard_normal() * signed_time
    # telegraph: exact event-driven integration across segment boundaries
    state = p.a_tls if rng.random() < 0.5 else -p.a_tls
    edges = [0.0]
    cur = rng.exponential(1.0 / p.gamma_tls)
    while cur < p.t:
        edges.append(cur)
        cur += rng.exponential(1.0 / p.gamma_tls)
    edges.append(p.t)
    s0 = state
    for a, b in zip(edges, edges[1:]):
        for sa, sb, sign in segs:
            lo, hi = max(a, sa), min(b, sb)
            if hi > lo:
                phi += sign * s0 * (hi - lo)
        s0 = -s0
    return float(phi)


def _apply_pauli(vec: np.ndarray, bit: int, which: int) -> None:
    """In-place X/Y/Z on one qubit of a basis-indexed vector."""
    idx = np.arange(len(vec))
    hi = (idx >> bit) & 1 == 1
    if which == 2:  # Z
        vec[hi] *= -1
        return
    lo_idx = idx[hi] - (1 << bit)
    tmp = vec[lo_idx].copy()
    vec[lo_idx] = vec[hi]
    vec[hi] = tmp
    if which == 1:  # Y = iXZ up to global phase: add the signs
        vec[hi] *= 1j
        vec[lo_idx] *= -1j


def noisy_vectors(phi: np.ndarray, n_data: int, p: MemoryParams, rng: np.random.Generator) -> np.ndarray:
    """(trajectories, dim) noisy copies of the stored state vector phi.

    Dephasing phases and damping jumps act on the n_data low bits only —
    any filter ancilla dimension rides along untouched (it is measured at
    readout, not stored).
    """
    dim = len(phi)
    idx = np.arange(dim)
    gamma = 1.0 - np.exp(-p.t / p.T1)
    out = np.empty((p.trajectories, dim), dtype=complex)
    for m in range(p.trajectories):
        vec = phi.astype(complex).copy()
        for transfer in range(2):  # swap-in before storage, swap-out after
            if p.transfer_infidelity > 0.0:
                for j in range(n_data):
                    if rng.random() < p.transfer_infidelity:
                        _apply_pauli(vec, j, int(rng.integers(3)))
            if transfer == 1:
                break
            for j in range(n_data):
                hi = (idx >> j) & 1 == 1
                vec[hi] *= np.exp(1j * sample_qubit_phase(p, rng))
                p1 = float(np.sum(np.abs(vec[hi]) ** 2))
                if p1 > 0.0 and rng.random() < gamma * p1:
                    lowered = np.zeros_like(vec)
                    lowered[idx[hi] - (1 << j)] = vec[hi]
                    vec = lowered / np.sqrt(p1)
                else:
                    vec[hi] *= np.sqrt(1.0 - gamma)
                    vec /= np.sqrt(1.0 - gamma * p1)
        out[m] = vec
    return out


def memory_distribution(
    phi: np.ndarray,
    n_data: int,
    p: MemoryParams,
    rng: np.random.Generator,
    mode: str,
    basis: str = "time",
) -> np.ndarray:
    """Outcome probabilities after storage, matching the circuit layouts:
    z: data bits (+ filter bit); homodyne: homodyne ancilla on top, with
    the fresh (unstored) uniform reference interfering against each
    noisy trajectory. Identical math to the readout circuits."""
    N = 1 << n_data
    vecs = noisy_vectors(phi, n_data, p, rng)
    if basis == "freq":
        blocks = vecs.reshape(p.trajectories, -1, N)
        vecs = (np.sqrt(N) * np.fft.ifft(blocks, axis=2)).reshape(p.trajectories, -1)
    if mode == "z":
        probs = np.mean(np.abs(vecs) ** 2, axis=0)
    else:
        ref = np.zeros(vecs.shape[1])
        ref[:N] = 1.0 / np.sqrt(N)  # reference lives in the filter=0 block
        plus = np.mean(np.abs((vecs + ref) / 2.0) ** 2, axis=0)
        minus = np.mean(np.abs((vecs - ref) / 2.0) ** 2, axis=0)
        probs = np.concatenate([plus, minus])
    probs = np.clip(probs, 0.0, None)
    return probs / probs.sum()


def measure_with_memory(
    circuit: QuantumCircuit,
    shots: int,
    mode: str,
    basis: str,
    n_data: int,
    p: MemoryParams,
    rng: np.random.Generator,
    evaluator: TapeEvaluator | None = None,
) -> np.ndarray:
    """Sample shot outcomes (uint16, standard bit layout) for one stored frame."""
    phi = evaluator.output_vector(circuit) if evaluator is not None else Statevector(circuit).data
    probs = memory_distribution(phi, n_data, p, rng, mode, basis)
    counts = rng.multinomial(shots, probs)
    outcomes = np.repeat(np.arange(len(probs), dtype=np.uint16), counts)
    rng.shuffle(outcomes)
    return outcomes
