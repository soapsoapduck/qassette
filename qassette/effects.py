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
from qiskit.circuit.library import DiagonalGate, QFTGate, UCRYGate


def build_fx_circuit(n: int, chain: tuple = ()) -> QuantumCircuit:
    """QFT -> chain gates -> IQFT on n data qubits. Empty chain == identity.

    Unitary effects act on the n data qubits (in the frequency basis).
    Post-selected effects (lowpass/highpass/fshift) act on n+1 qubits,
    where qubit n is the filter ancilla; the whole fx circuit then has
    n+1 qubits and downstream code post-selects on ancilla = 0.
    """
    width = n
    for effect in chain:
        if isinstance(effect, QuantumCircuit) and effect.num_qubits > n:
            width = n + 1
    qc = QuantumCircuit(width, name="fx")
    qc.append(QFTGate(n), range(n))
    for effect in chain:
        if isinstance(effect, QuantumCircuit):
            qc.compose(effect, range(effect.num_qubits), inplace=True)
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


def _postselect_ucry(n: int, reject: np.ndarray, angle: float, name: str) -> QuantumCircuit:
    """n+1 qubit circuit rotating the filter ancilla (qubit n) by `angle`
    for every rejected stored frequency index. Post-select ancilla = 0:
    rejected bins keep amplitude cos(angle/2) (0 for angle = pi)."""
    from qiskit import transpile

    angles = [float(angle) if reject[k] else 0.0 for k in range(2**n)]
    qc = QuantumCircuit(n + 1, name=name)
    qc.append(UCRYGate(angles), [n, *range(n)])
    # flatten to plain gates: QPY-serializable and safe for to_gate()/control
    flat = transpile(qc, basis_gates=["cx", "u"], optimization_level=0)
    flat.name = name
    return flat


def fx_lowpass(n: int, cutoff: int, angle: float = np.pi) -> QuantumCircuit:
    """Post-selected lowpass: bins with |k_signed| > cutoff are rotated into
    the filter ancilla and discarded on ancilla = 0. The mask is real and
    even in +-k, so real inputs stay real. `angle` < pi gives a soft
    roll-off: rejected bins survive with amplitude cos(angle/2) instead
    of being zeroed. Post-selection success = kept-energy fraction."""
    N = 2**n
    ks = _signed(np.arange(N), N)
    tag = f"lowpass:{cutoff}" if angle == np.pi else f"lowpass:{cutoff}:{angle:g}"
    return _postselect_ucry(n, np.abs(ks) > cutoff, angle, tag)


def fx_highpass(n: int, cutoff: int, angle: float = np.pi) -> QuantumCircuit:
    """Post-selected highpass: rejects bins with |k_signed| < cutoff.
    See fx_lowpass for the ancilla/softness semantics."""
    N = 2**n
    ks = _signed(np.arange(N), N)
    tag = f"highpass:{cutoff}" if angle == np.pi else f"highpass:{cutoff}:{angle:g}"
    return _postselect_ucry(n, np.abs(ks) < cutoff, angle, tag)


def fx_fshift(n: int, k: int) -> QuantumCircuit:
    """Single-sideband frequency shifter: signal frequencies move up by
    k * sr / N Hz. Positive-frequency bins get +k and negative bins -k,
    implemented on the stored index register as a Draper adder over the
    low n-1 bits (QFT, phase ramp, inverse QFT) whose direction is
    CONTROLLED ON THE SIGN BIT (qubit n-1): an unconditional +k ramp plus
    a sign-controlled -2k ramp.

    Wraparound/collisions: bins within k of Nyquist would wrap across the
    +-N/2 boundary and alias, and DC (k_signed = 0) has no mirror partner
    after the shift, which would break conjugate symmetry. All of these —
    |k_signed| >= N/2 - k, plus DC — are zeroed via the filter ancilla
    (UCRY(pi) + post-selection) BEFORE the shift, so surviving bins never
    change sign bit and never collide."""
    N = 2**n
    M = N // 2
    if not 0 < k < M:
        raise ValueError(f"fshift amount must be in (0, {M}), got {k}")
    ks = _signed(np.arange(N), N)
    reject = (np.abs(ks) >= M - k) | (ks == 0)
    qc = QuantumCircuit(n + 1, name=f"fshift:{k}")
    qc.compose(_postselect_ucry(n, reject, np.pi, "guard"), range(n + 1), inplace=True)
    qc.append(QFTGate(n - 1), range(n - 1))
    for j in range(n - 1):
        theta = 2 * np.pi * k * (2**j) / M
        qc.p(theta, j)
        qc.cp(-2 * theta, n - 1, j)
    qc.append(QFTGate(n - 1).inverse(), range(n - 1))
    return qc


def uses_ancilla(fx_circuit: QuantumCircuit, n: int) -> bool:
    """True when the fx circuit carries a filter ancilla (post-selection)."""
    return fx_circuit.num_qubits > n


def parse_fx(spec: str, n: int) -> tuple:
    """Parse an effect-chain spec like "delay:3,chirp:0.5,ringmod:4,lowpass:40:2.5".

    At most ONE post-selected effect (lowpass/highpass/fshift) per chain:
    they share the single filter ancilla, and a second UCRY would rotate
    already-rejected amplitude back into the kept branch.
    """
    if not spec or not spec.strip():
        return ()
    chain = []
    filters = 0
    for token in spec.split(","):
        name, *args = [t.strip() for t in token.strip().split(":")]
        try:
            if name == "delay":
                (d,) = args
                chain.append(fx_delay(n, int(d)))
            elif name == "chirp":
                (c,) = args
                chain.append(fx_chirp(n, float(c)))
            elif name == "ringmod":
                (k,) = args
                chain.append(fx_ringmod(n, int(k)))
            elif name in ("lowpass", "highpass"):
                cutoff = int(args[0])
                angle = float(args[1]) if len(args) > 1 else float(np.pi)
                if not 0 < angle <= np.pi:
                    raise ValueError(f"softness angle must be in (0, pi], got {angle}")
                build = fx_lowpass if name == "lowpass" else fx_highpass
                chain.append(build(n, cutoff, angle))
                filters += 1
            elif name == "fshift":
                (k,) = args
                chain.append(fx_fshift(n, int(k)))
                filters += 1
            else:
                raise KeyError
        except KeyError:
            raise ValueError(
                f"unknown effect {name!r}; expected one of "
                "['chirp', 'delay', 'fshift', 'highpass', 'lowpass', 'ringmod']"
            ) from None
        except (TypeError, ValueError) as e:
            if "softness angle" in str(e) or "fshift amount" in str(e):
                raise ValueError(str(e)) from None
            raise ValueError(f"bad argument for {name!r}: {token!r} ({e})") from e
    if filters > 1:
        raise ValueError("at most one post-selected effect (lowpass/highpass/fshift) per chain")
    return tuple(chain)
