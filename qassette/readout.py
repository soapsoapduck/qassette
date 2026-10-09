"""Exact (statevector) readout of encoded frames. Simulation only."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

_IMAG_TOL = 1e-9


def read_exact(circuit: QuantumCircuit) -> np.ndarray:
    """Return the real amplitude vector of the circuit's output state.

    Index k of the result is computational-basis index k (little-endian:
    bit j of k is qubit j), matching the sample index used at encode time.
    """
    amps = Statevector(circuit).data
    assert np.max(np.abs(amps.imag)) < _IMAG_TOL, "state has non-real amplitudes"
    return amps.real.copy()
