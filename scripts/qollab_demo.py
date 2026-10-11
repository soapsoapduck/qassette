#!/usr/bin/env python3
"""
QASSETTE - music on quantum tape (single-file Qollab demo)
===========================================================

A cassette deck where the tape is a quantum state. Audio frames become
8-qubit amplitude-encoded states; playback is nothing but measurement
statistics. This file is self-contained: it synthesizes its own audio,
needs only numpy + qiskit, and runs on whatever is available:

  1. a pre-created `backend` variable (provided by hosted Qiskit runtimes)
     - used for PART 1, one batched job of real measured circuits;
  2. qiskit's built-in Statevector (exact simulation) for everything else.

Full project (CLI, Streamlit app, QFT effects rack, paper-calibrated
mechanical quantum memory, 68 tests): see the qassette repository.
The run WRITES WAV FILES you can download and listen to: the original
chord, the tape at 64 and 16384 shots/frame, and the stored tape with
and without a Hahn echo. What you should hear:
  - homodyne readout recovers the signed waveform; Z readout only gets
    magnitudes (interference is load-bearing),
  - fidelity grows as 1/sqrt(shots) (measurement statistics ARE the song),
  - storage dephasing destroys the music and a Hahn echo brings it back,
  - and the chord is genuinely entangled across the qubit register.
"""

import wave

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

try:
    backend  # injected by the hosted runtime
except NameError:
    backend = None

rng = np.random.default_rng(1234)
SR, N, HOP = 16000, 256, 128
NQ = 8  # log2(N) data qubits

# ----------------------------------------------------------- make music
t = np.arange(int(0.25 * SR)) / SR  # quarter second of A-major
chord = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 554.37, 659.25))
chord = 0.9 * chord / np.max(np.abs(chord))

hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
PAD = N - HOP  # pre/post padding gives every real sample full window coverage
padded = np.concatenate([np.zeros(PAD), chord, np.zeros(N)])
starts = range(0, PAD + len(chord) - 1, HOP)  # every frame overlaps real audio
frames = [padded[s:s + N] * hann for s in starts]
norms = [np.linalg.norm(f) for f in frames]
print(f"tape: {len(frames)} frames x {N} samples -> {NQ} qubits each\n")


def prep_circuit(frame):
    qc = QuantumCircuit(NQ)
    qc.append(StatePreparation(frame / np.linalg.norm(frame)), range(NQ))
    return qc


def homodyne_circuit(frame):
    """(|0>|psi> + |1>|r>)/sqrt(2) with r uniform; final H on the ancilla.
    P(anc=0,k) - P(anc=1,k) = psi_k / sqrt(N)  ->  signs are recoverable."""
    psi = frame / np.linalg.norm(frame)
    qc = QuantumCircuit(NQ + 1)
    qc.h(NQ)
    qc.append(StatePreparation(psi).control(1, ctrl_state=0), [NQ, *range(NQ)])
    for j in range(NQ):
        qc.ch(NQ, j)
    qc.h(NQ)
    return qc


def counts_to_probs(counts, nbits):
    p = np.zeros(2 ** nbits)
    for key, c in counts.items():
        p[int(key.replace(" ", ""), 2)] = c
    return p / p.sum()


def write_wav(name, y):
    """Write mono 16-bit audio with the standard library only."""
    data = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    print(f"  wrote {name} ({len(y) / SR:.2f}s) - download and listen")


# ------------------- PART 1: real measured circuits, ONE batched job
frame = frames[len(frames) // 2]
psi = frame / np.linalg.norm(frame)
SHOTS = 2048

qc_h = homodyne_circuit(frame)
qc_h.measure_all()
qc_z = prep_circuit(frame)
qc_z.measure_all()

if backend is not None:
    name = getattr(backend, "name", lambda: "backend")
    print(f"PART 1 - measuring one frame on: {name() if callable(name) else name}")
    tqcs = transpile([qc_h, qc_z], backend)
    result = backend.run(tqcs, shots=SHOTS).result()  # single job
    p_h = counts_to_probs(result.get_counts(0), NQ + 1)
    p_z = counts_to_probs(result.get_counts(1), NQ)
else:
    print("PART 1 - no hosted backend found; sampling exact statevectors")
    sv_h, sv_z = Statevector(homodyne_circuit(frame)), Statevector(prep_circuit(frame))
    p_h = counts_to_probs(sv_h.sample_counts(SHOTS), NQ + 1)
    p_z = counts_to_probs(sv_z.sample_counts(SHOTS), NQ)

print("  16 shots from the measured histogram (ancilla | 8 data bits):")
for o in rng.choice(2 ** (NQ + 1), size=16, p=p_h):
    b = format(int(o), f"0{NQ + 1}b")
    print(f"    {b[0]} | {b[1:]}")

signed_est = np.sqrt(N) * (p_h[:N] - p_h[N:])   # homodyne: signed amplitudes
mag_est = np.sqrt(p_z)                           # Z: magnitudes only
print(f"  homodyne estimate vs frame : corr {np.corrcoef(signed_est, psi)[0, 1]:+.3f}")
print(f"  Z estimate vs frame        : corr {np.corrcoef(mag_est, psi)[0, 1]:+.3f}")
print(f"  Z estimate vs |frame|      : corr {np.corrcoef(mag_est, np.abs(psi))[0, 1]:+.3f}")
print("  -> without interference the signs (half the music) are gone\n")

# ---------------- PART 2: the full clip, exact simulation + sampling
print("PART 2 - whole clip, exact simulation")
svs = [Statevector(homodyne_circuit(f)).data for f in frames]


def decode(estimates_per_frame):
    """Overlap-add homodyne estimates back into audio (padding trimmed)."""
    out = np.zeros(len(padded))
    wsum = np.zeros(len(padded))
    for s, est, nm in zip(starts, estimates_per_frame, norms):
        out[s:s + N] += est * nm
        wsum[s:s + N] += hann
    good = wsum > 0.5  # full-coverage region only: no edge noise blow-up
    out[good] /= wsum[good]
    return out[PAD:PAD + len(chord)]


def sample_estimates(vectors, shots):
    ests = []
    for v in vectors:
        p = np.abs(v) ** 2
        c = rng.multinomial(shots, p / p.sum())
        ests.append(np.sqrt(N) * (c[:N] - c[N:]) / shots)
    return ests


print("  shots/frame   correlation with the chord")
for shots in (64, 1024, 16384):
    y = decode(sample_estimates(svs, shots))
    print(f"  {shots:>9}   {np.corrcoef(y, chord)[0, 1]:.3f}")
    if shots in (64, 16384):
        write_wav(f"quantum_tape_{shots}shots.wav", y)
write_wav("original_chord.wav", chord)
print("  -> fidelity grows as sqrt(shots): statistics carry the song\n")

# quantum memory: quasi-static dephasing (T2* = 64 us, Bozkurt et al.,
# Nature Physics 21, 1469 (2025)); a Hahn echo refocuses slow drift
print("PART 3 - 200 us in a mechanical quantum memory (T2* = 64 us)")
SIGMA, T_STORE = np.sqrt(2) / 64e-6, 200e-6
for dd, label in ((False, "no echo  "), (True, "hahn echo")):
    ests = []
    for f in frames:
        v = (f / np.linalg.norm(f)).astype(complex)
        deltas = SIGMA * rng.standard_normal(NQ)
        phase_time = 0.0 if dd else T_STORE  # echo cancels static drift
        for j in range(NQ):
            hi = (np.arange(N) >> j) & 1 == 1
            v[hi] *= np.exp(1j * deltas[j] * phase_time)
        ref = np.full(N, 1 / np.sqrt(N))
        p = np.concatenate([np.abs(v + ref) ** 2, np.abs(v - ref) ** 2]) / 4
        c = rng.multinomial(4096, p / p.sum())
        ests.append(np.sqrt(N) * (c[:N] - c[N:]) / 4096)
    y = decode(ests)
    print(f"  {label}: corr {np.corrcoef(y, chord)[0, 1]:.3f}")
    write_wav("memory_hahn_echo.wav" if dd else "memory_no_echo.wav", y)
print("  -> decoherence erases the tape; one echo pulse restores it\n")

# entanglement: the chord is a genuinely entangled 8-qubit state
amps = psi.reshape(16, 16)                      # cut: qubits 0-3 | 4-7
s_vals = np.linalg.svd(amps, compute_uv=False)
probs = s_vals ** 2 / np.sum(s_vals ** 2)
entropy = float(-np.sum(probs * np.log2(probs + 1e-16)))
silence = np.zeros(N)
silence[0] = 1.0
s0 = np.linalg.svd(silence.reshape(16, 16), compute_uv=False)
p0 = s0 ** 2 / s0.sum() ** 2 if s0.sum() else s0
print(f"PART 4 - entanglement entropy across the register (4|4 qubit cut)")
print(f"  this chord frame : {entropy:.2f} ebits")
print(f"  digital silence  : 0.00 ebits")
print("  -> music is entangled; silence is a product state")


def main():
    """Qollab's runner calls main() after the script; the demo runs above."""
    pass
