"""Readout of encoded frames: exact statevector, sampled Z-basis, and an
ancilla-interference ("homodyne") measurement that recovers signs.
Simulation only (Statevector sampling by default, AerSimulator behind a flag).

Bit conventions (Qiskit little-endian): data qubits are 0..n-1, the homodyne
ancilla is qubit n. An integer outcome o therefore splits as
k = o & (N - 1) (data index) and anc = o >> n; in a printed bitstring the
ancilla is the LEFTMOST character.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

_IMAG_TOL = 1e-9

MODES = ("exact", "z", "homodyne")


def read_exact(circuit: QuantumCircuit) -> np.ndarray:
    """Return the real amplitude vector of the circuit's output state.

    Index k of the result is computational-basis index k (little-endian:
    bit j of k is qubit j), matching the sample index used at encode time.
    """
    amps = Statevector(circuit).data
    if np.max(np.abs(amps.imag)) >= _IMAG_TOL:
        raise ValueError("state has non-real amplitudes")
    return amps.real.copy()


def build_homodyne_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Interference circuit on n+1 qubits (no measurements) for sign readout.

    The ENTIRE input circuit (state preparation plus any effect gates) is
    anti-controlled on the ancilla, so the anc=0 branch holds the circuit's
    actual output state |psi>, not just its initially prepared one. With r
    the uniform state (controlled-H on every data qubit, r_k = 1/sqrt(N)):

        H(anc); anti-controlled circuit-gate on anc=0;
        controlled-H on every data qubit on anc=1

    produces (|0>|psi> + |1>|r>)/sqrt(2). The final H on the ancilla maps
    |0> -> (|0>+|1>)/sqrt(2) and |1> -> (|0>-|1>)/sqrt(2), giving

        (1/2) [ |0>(|psi> + |r>) + |1>(|psi> - |r>) ]

    so

        P(anc=0, k) - P(anc=1, k)
            = (1/4)(|psi_k + r_k|^2 - |psi_k - r_k|^2)
            = Re(psi_k) * r_k = Re(psi_k) / sqrt(N)

    hence Re(psi_k) = sqrt(N) * (P(anc=0, k) - P(anc=1, k)). Note only the
    real quadrature is read: effects that rotate amplitudes into the
    imaginary plane are invisible to this estimator (keep effects real, or
    add an S-dagger on the ancilla before its final H to read Im(psi_k)).
    """
    n = circuit.num_qubits
    if circuit.num_clbits or any(
        inst.operation.name in ("measure", "reset") for inst in circuit.data
    ):
        raise ValueError("homodyne readout requires a measurement-free unitary circuit")
    qc = QuantumCircuit(n + 1)
    qc.h(n)
    if circuit.data:  # empty circuit (zero frame) prepares |0...0>; skip identity gate
        qc.append(circuit.to_gate(label="state").control(1, ctrl_state=0), [n, *range(n)])
    for j in range(n):
        qc.ch(n, j)
    qc.h(n)
    return qc


def homodyne_statevector(circuit: QuantumCircuit) -> Statevector:
    """Statevector of the homodyne interference circuit for an encode circuit."""
    return Statevector(build_homodyne_circuit(circuit))


def sample_statevector(sv: Statevector, shots: int, seed: int | None = None) -> np.ndarray:
    """Sample Z-basis outcomes from a statevector as little-endian uint16 ints."""
    sv.seed(seed)
    mem = sv.sample_memory(shots)
    return np.fromiter((int(s, 2) for s in mem), dtype=np.uint16, count=len(mem))


def _aer_sample(circuit: QuantumCircuit, shots: int, seed: int | None = None) -> np.ndarray:
    """Sample by running the circuit (plus measurements) on AerSimulator."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = circuit.copy()
    qc.measure_all()
    sim = AerSimulator()
    result = sim.run(transpile(qc, sim), shots=shots, memory=True, seed_simulator=seed).result()
    mem = result.get_memory()
    return np.fromiter((int(s, 2) for s in mem), dtype=np.uint16, count=len(mem))


def sample_z(
    circuit: QuantumCircuit,
    shots: int,
    *,
    use_aer: bool = False,
    seed: int | None = None,
    basis: str = "time",
) -> np.ndarray:
    """Z-basis shot outcomes of the data register (n-bit integers).

    basis="freq" appends a QFTGate before measuring, so outcomes are
    frequency bins (undoing any trailing IQFT of the effect slot: the
    state is measured right after the QFT). See effects.py for the
    index convention; magnitudes match numpy's after folding k / N-k.
    """
    if basis not in ("time", "freq"):
        raise ValueError(f"basis must be 'time' or 'freq', got {basis!r}")
    if basis == "freq":
        from qiskit.circuit.library import QFTGate

        circuit = circuit.copy()
        circuit.append(QFTGate(circuit.num_qubits), range(circuit.num_qubits))
    if use_aer:
        return _aer_sample(circuit, shots, seed)
    return sample_statevector(Statevector(circuit), shots, seed)


def sample_homodyne(circuit: QuantumCircuit, shots: int, *, use_aer: bool = False, seed: int | None = None) -> np.ndarray:
    """Homodyne shot outcomes ((n+1)-bit integers, ancilla in bit n)."""
    hqc = build_homodyne_circuit(circuit)
    if use_aer:
        return _aer_sample(hqc, shots, seed)
    return sample_statevector(Statevector(hqc), shots, seed)


def estimate_z(outcomes: np.ndarray, N: int) -> np.ndarray:
    """|x_k| estimates from Z-basis outcomes: sqrt(count_k / shots). Signs are lost.

    sqrt of the count fraction is biased low for small-probability bins
    (Jensen's inequality), so treat these as display/listening magnitudes,
    not precision estimates.
    """
    counts = np.bincount(outcomes, minlength=N)
    return np.sqrt(counts / len(outcomes))


def estimate_homodyne(outcomes: np.ndarray, N: int) -> np.ndarray:
    """Signed Re(psi_k) estimates: sqrt(N) * (P(anc=0,k) - P(anc=1,k))."""
    counts = np.bincount(outcomes, minlength=2 * N)
    p = counts / len(outcomes)
    return np.sqrt(N) * (p[:N] - p[N:])


def read_frame(
    circuit: QuantumCircuit,
    mode: str = "exact",
    shots: int = 4096,
    *,
    use_aer: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Read one encoded frame back as an amplitude-vector estimate."""
    if mode not in MODES:
        raise ValueError(f"unknown readout mode {mode!r}; expected one of {MODES}")
    N = 2**circuit.num_qubits
    if mode == "exact":
        return read_exact(circuit)
    if mode == "z":
        return estimate_z(sample_z(circuit, shots, use_aer=use_aer, seed=seed), N)
    return estimate_homodyne(sample_homodyne(circuit, shots, use_aer=use_aer, seed=seed), N)
