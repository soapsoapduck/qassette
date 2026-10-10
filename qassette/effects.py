"""Frequency-domain effect chain for qassette.

Effects run inside a QFT sandwich: QFT -> [effect gates] -> inverse QFT,
so each effect acts on the frame's spectrum and the circuit returns to the
time basis afterwards. An empty chain is the identity (QFT then IQFT).

Qiskit QFTGate convention (verified against numpy): applying QFTGate(n) to
a state psi gives, at computational-basis index k,

    (QFT psi)_k = (1/sqrt(N)) * sum_j exp(+2j*pi*j*k/N) psi_j
                = conj(np.fft.fft(psi))[k] / sqrt(N)
                = sqrt(N) * np.fft.ifft(psi)[k]

i.e. Qiskit's exponent sign is opposite to np.fft.fft. For REAL frames
this means numpy's frequency bin k appears at index (N - k) mod N — the
frequency axis is negated. Magnitudes are unchanged, and folding bin k
with bin N-k (as any real-signal spectrogram does) erases the distinction.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate


def build_fx_circuit(n: int, chain: tuple = ()) -> QuantumCircuit:
    """QFT -> chain gates -> IQFT on n qubits. Empty chain == identity.

    Each chain entry is a Gate or QuantumCircuit acting on all n data
    qubits (in the frequency basis).
    """
    qc = QuantumCircuit(n, name="fx")
    qc.append(QFTGate(n), range(n))
    for effect in chain:
        if isinstance(effect, QuantumCircuit):
            qc.compose(effect, range(n), inplace=True)
        else:
            qc.append(effect, range(n))
    qc.append(QFTGate(n).inverse(), range(n))
    return qc


def chain_names(chain: tuple) -> list[str]:
    """Human-readable effect names for tape headers."""
    return [getattr(e, "name", type(e).__name__) for e in chain]
