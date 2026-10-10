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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, QFTGate


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


def _signed(k: np.ndarray, N: int) -> np.ndarray:
    """Map register index to signed frequency in [-N/2, N/2)."""
    return ((k + N // 2) % N) - N // 2


def fx_delay(n: int, d: int) -> QuantumCircuit:
    """Circular time shift by d samples: y[n] = x[(n - d) mod N] (np.roll(x, d)).

    In numpy terms Y_m = X_m * exp(-2j*pi*m*d/N); the register stores X at
    the negated index (see module docstring), so the stored-index-k phase is
    exp(+2j*pi*k*d/N) — one phase gate per qubit, P(2*pi*d*2^j/N) on qubit j.
    """
    qc = QuantumCircuit(n, name=f"delay:{d}")
    N = 2**n
    for j in range(n):
        qc.p(2 * np.pi * d * (2**j) / N, j)
    return qc


def fx_chirp(n: int, c: float) -> QuantumCircuit:
    """All-pass dispersion ("chirp"): phase exp(+1j*pi*c*k*|k|/N) at signed
    numpy frequency k in [-N/2, N/2), via DiagonalGate.

    On non-negative frequencies this is exactly exp(1j*pi*c*k^2/N); negative
    frequencies carry the CONJUGATE phase. The magnitude |phase argument| is
    symmetric in +-k, but the sign must be odd (conjugate symmetry): a fully
    even phase would NOT keep real inputs real — realness requires
    H(-k) = conj(H(k)). The Nyquist bin (k = -N/2, which has no +N/2
    partner) is left untouched for the same reason.

    The register stores X at the negated index, so the diagonal applied at
    stored index k uses the opposite sign: exp(-1j*pi*c*ks*|ks|/N).
    """
    N = 2**n
    ks = _signed(np.arange(N), N)
    diag = np.exp(-1j * np.pi * c * ks * np.abs(ks) / N)
    diag[N // 2] = 1.0  # Nyquist: keep real-input realness
    qc = QuantumCircuit(n, name=f"chirp:{c:g}")
    qc.append(DiagonalGate(list(diag)), range(n))
    return qc


def fx_ringmod(n: int, k: int) -> QuantumCircuit:
    """Ring modulation: add the constant k (mod N) to the frequency index
    register with a Draper-style phase adder (QFT on the register, one
    phase gate per qubit, inverse QFT).

    Shifting the stored spectrum X'_m = X_{m+k} multiplies the time signal
    by a pure per-sample phase: y[n] = x[n] * exp(-2j*pi*k*n/N).

    Why Z readout cannot see it: Z-basis statistics are |y[n]|^2, and
    |exp(i*theta)| = 1, so every bin probability is unchanged. Homodyne
    readout interferes y against a fixed real reference and estimates the
    REAL quadrature, Re(y[n]) = x[n] * cos(2*pi*k*n/N) — the modulation
    becomes audible as a classic ring-mod voicing at k * sr / N Hz.
    """
    qc = QuantumCircuit(n, name=f"ringmod:{k}")
    N = 2**n
    qc.append(QFTGate(n), range(n))
    for j in range(n):
        qc.p(2 * np.pi * k * (2**j) / N, j)
    qc.append(QFTGate(n).inverse(), range(n))
    return qc


def parse_fx(spec: str, n: int) -> tuple:
    """Parse an effect-chain spec like "delay:3,chirp:0.5,ringmod:4"."""
    if not spec or not spec.strip():
        return ()
    builders = {"delay": (fx_delay, int), "chirp": (fx_chirp, float), "ringmod": (fx_ringmod, int)}
    chain = []
    for token in spec.split(","):
        name, _, arg = token.strip().partition(":")
        if name not in builders:
            raise ValueError(f"unknown effect {name!r}; expected one of {sorted(builders)}")
        build, cast = builders[name]
        try:
            chain.append(build(n, cast(arg)))
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad argument for {name!r}: {arg!r} ({e})") from e
    return tuple(chain)
