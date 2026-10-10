"""Amplitude encoding of audio frames into quantum states.

A length-N frame (N a power of two) is L2-normalized and prepared as the
amplitude vector of an n = log2(N) qubit state. Qiskit is little-endian,
but StatePreparation takes the amplitude vector indexed exactly like
Statevector: sample index k lands at computational-basis index k, where
bit j of k is the state of qubit j.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from .audio_io import frame_signal
from .effects import build_fx_circuit


def encode_frame(frame: np.ndarray, fx: QuantumCircuit | None = None) -> tuple[QuantumCircuit, float]:
    """Encode one frame as a quantum state; returns (circuit, norm).

    norm is the L2 norm of the frame; norm == 0.0 flags an all-zero frame,
    which is encoded as |0...0> and decodes back to silence. If fx is
    given, it is composed after the preparation (the effect-chain slot).
    """
    frame = np.asarray(frame, dtype=np.float64).ravel()
    N = len(frame)
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("frame length must be a power of two")
    n = N.bit_length() - 1
    qc = QuantumCircuit(n)
    norm = float(np.linalg.norm(frame))
    if norm != 0.0:
        qc.append(StatePreparation(frame / norm), range(n))
    if fx is not None:
        qc.compose(fx, range(n), inplace=True)
    return qc, norm


def encode_signal(
    x: np.ndarray, N: int = 256, hop: int = 128, chain: tuple = ()
) -> list[tuple[QuantumCircuit, float]]:
    """Frame x (windowed, via frame_signal) and amplitude-encode every frame.

    Every circuit carries the QFT -> chain -> IQFT effect slot (identity
    for an empty chain), so tapes always contain the effect chain.
    """
    fx = build_fx_circuit(N.bit_length() - 1, chain)
    return [encode_frame(f, fx=fx) for f in frame_signal(x, N=N, hop=hop)]
