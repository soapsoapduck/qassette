"""Render every demo asset into outputs/demo/, reproducibly (fixed seeds).

Produces the shots ladder, readout ablation, one WAV per QFT effect, the
memory decay/restoration set, the coherent-vs-memoryless dephasing
ablation, the raw-bits text panel, both spectrograms, and ablations.md
(metrics + the quantum principle each ablation demonstrates).
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qassette.audio_io import load_audio, save_audio  # noqa: E402
from qassette.decode import frames_to_audio  # noqa: E402
from qassette.effects import parse_fx  # noqa: E402
from qassette.encode import encode_signal  # noqa: E402
from qassette.formats import Qbits  # noqa: E402, F401
from qassette.formats import to_ascii  # noqa: E402
from qassette.memory import (  # noqa: E402
    MemoryParams,
    TapeEvaluator,
    _dd_boundaries,
    measure_with_memory,
)
from qassette.readout import (  # noqa: E402
    estimate_homodyne,
    estimate_z,
    homodyne_statevector,
    sample_statevector,
    sample_z,
)
from qassette.viz import quantum_spectrogram  # noqa: E402

OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
SR, N, HOP, NQ = 16000, 256, 128, 8
CHORD = ROOT / "samples" / "a_major_chord_2s.wav"
SWEEP = ROOT / "samples" / "sweep_100_4000hz_3s.wav"

SHOT_LADDER = (1, 16, 256, 4096, 65536)
FX_SPECS = ("delay:64", "chirp:4", "ringmod:16", "lowpass:8", "fshift:8")
FX_SHOTS = 16384
# decay + restoration set (Check C of the memory audit)
MEMORY_SET = [
    (0.0, "none"), (200e-6, "none"), (200e-6, "hahn"),
    (500e-6, "none"), (500e-6, "hahn"), (500e-6, "cpmg"),
    (1e-3, "none"), (1e-3, "hahn"), (1e-3, "cpmg"),
]

metrics: dict[str, float] = {}


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def _save(name, y):
    save_audio(OUT / name, np.clip(y, -1, 1), SR)
    print(f"  wrote {name}")


def main():
    t_start = time.perf_counter()
    x = load_audio(CHORD)
    encoded = encode_signal(x, N=N, hop=HOP)
    norms = [nm for _, nm in encoded]

    # ---- A: shots ladder (one homodyne statevector per frame, 5 ladders)
    print("A: shots ladder")
    svs = [homodyne_statevector(qc) for qc, _ in encoded]
    for shots in SHOT_LADDER:
        amps = [
            estimate_homodyne(sample_statevector(sv, shots, seed=10_000 + 7 * shots + i), N)
            for i, sv in enumerate(svs)
        ]
        y = frames_to_audio(amps, norms, hop=HOP, length=len(x))
        metrics[f"shots_{shots}"] = _corr(y, x)
        _save(f"shots_{shots}.wav", y)

    # ---- B: readout ablation (z has no reference arm: signs lost)
    print("B: readout ablation")
    z_rows = [sample_z(qc, 4096, seed=20_000 + i) for i, (qc, _) in enumerate(encoded)]
    y_z = frames_to_audio([estimate_z(r, N) for r in z_rows], norms, hop=HOP, length=len(x))
    metrics["readout_z"] = _corr(y_z, x)
    metrics["readout_z_abs"] = _corr(y_z, np.abs(x))
    _save("readout_z.wav", y_z)
    amps = [
        estimate_homodyne(sample_statevector(sv, 4096, seed=21_000 + i), N)
        for i, sv in enumerate(svs)
    ]
    y_h = frames_to_audio(amps, norms, hop=HOP, length=len(x))
    metrics["readout_homodyne"] = _corr(y_h, x)
    _save("readout_homodyne.wav", y_h)

    # ---- C: one WAV per QFT effect
    print("C: effects")
    for spec in FX_SPECS:
        name = spec.split(":")[0]
        chain = parse_fx(spec, NQ)
        enc_fx = encode_signal(x, N=N, hop=HOP, chain=chain)
        filtered = enc_fx[0][0].num_qubits > NQ
        fb = NQ if filtered else None
        amps = []
        for i, (qc, _) in enumerate(enc_fx):
            sv = homodyne_statevector(qc, data_qubits=NQ)
            outs = sample_statevector(sv, FX_SHOTS, seed=30_000 + 13 * len(name) + i)
            amps.append(estimate_homodyne(outs, N, filter_bit=fb))
        y = frames_to_audio(amps, [nm for _, nm in enc_fx], hop=HOP, length=len(x))
        metrics[f"fx_{name}"] = _corr(y, x)
        _save(f"fx_{name}.wav", y)

    # ---- D: memory decay + restoration (fast operator-cached path)
    print("D: memory set")
    ev = TapeEvaluator()
    for t, dd in MEMORY_SET:
        amps = []
        for i, (qc, _) in enumerate(encoded):
            outs = measure_with_memory(
                qc, FX_SHOTS, "homodyne", "time", NQ,
                MemoryParams(t=t, dd=dd), np.random.default_rng(40_000 + i), ev,
            )
            amps.append(estimate_homodyne(outs, N))
        y = frames_to_audio(amps, norms, hop=HOP, length=len(x))
        tag = f"memory_{int(t * 1e6):04d}us_{dd}"
        metrics[tag] = _corr(y, x)
        _save(tag + ".wav", y)

    # ---- E: dephasing ablation — coherent noise vs memoryless, both hahn
    print("E: dephasing ablation")
    t_ab = 200e-6
    phis = [ev.output_vector(qc) for qc, _ in encoded]

    def markov_homodyne(pv, p, rng, t2):
        """Memoryless dephasing: independent Gaussian phase per dd segment
        with variance 2*tau/T2 — an echo cannot refocus it."""
        vecs = np.empty((p.trajectories, len(pv)), dtype=complex)
        idx = np.arange(len(pv))
        for m in range(p.trajectories):
            vec = pv.astype(complex).copy()
            for j in range(NQ):
                ph = sum(
                    rng.standard_normal() * np.sqrt(2 * (b - a) / t2)
                    for a, b, _s in _dd_boundaries(p.t, p.dd)
                )
                vec[(idx >> j) & 1 == 1] *= np.exp(1j * ph)
            vecs[m] = vec
        ref = np.full(len(pv), 1 / np.sqrt(N))
        pr = np.concatenate([
            np.mean(np.abs((vecs + ref) / 2) ** 2, axis=0),
            np.mean(np.abs((vecs - ref) / 2) ** 2, axis=0),
        ])
        pr = np.clip(pr, 0, None)
        return pr / pr.sum()

    for kind in ("coherent", "memoryless"):
        amps = []
        for i, pv in enumerate(phis):
            rng = np.random.default_rng(50_000 + i)
            p = MemoryParams(t=t_ab, dd="hahn")
            if kind == "coherent":
                probs_vec = None
                outs = measure_with_memory(
                    encoded[i][0], FX_SHOTS, "homodyne", "time", NQ, p, rng, ev
                )
            else:
                pr = markov_homodyne(pv, p, rng, 64e-6)
                counts = rng.multinomial(FX_SHOTS, pr)
                outs = np.repeat(np.arange(len(pr), dtype=np.uint16), counts)
            amps.append(estimate_homodyne(outs, N))
        y = frames_to_audio(amps, norms, hop=HOP, length=len(x))
        metrics[f"dephasing_{kind}_hahn"] = _corr(y, x)
        _save(f"dephasing_{kind}_hahn.wav", y)

    # ---- F: bits panel + spectrograms
    print("F: bits + spectrograms")
    rows = np.stack([sample_statevector(svs[i], 4096, seed=60_000 + i) for i in range(len(svs))])
    qb = Qbits(
        header={"version": 1, "sr": SR, "N": N, "hop": HOP, "length": len(x),
                "mode": "homodyne", "basis": "time", "shots": 4096,
                "norms": norms, "effects": [], "filter": False},
        outcomes=rows,
    )
    (OUT / "bits_frame10.txt").write_text(to_ascii(qb, frame=10, max_shots=32), encoding="utf-8")
    print("  wrote bits_frame10.txt")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sps

    xs = load_audio(SWEEP)
    f2, t2, s2 = sps.spectrogram(xs, fs=SR, window="hann", nperseg=N, noverlap=N - HOP)
    enc_s = encode_signal(xs, N=N, hop=HOP)
    srows = np.stack([
        sample_z(qc, 4096, seed=70_000 + i, basis="freq") for i, (qc, _) in enumerate(enc_s)
    ])
    qbs = Qbits(
        header={"version": 1, "sr": SR, "N": N, "hop": HOP, "length": len(xs),
                "mode": "z", "basis": "freq", "shots": 4096,
                "norms": [nm for _, nm in enc_s], "effects": [], "filter": False},
        outcomes=srows,
    )
    fq, tq, pq = quantum_spectrogram(qbs)
    for fname, (f, t, s, title) in {
        "spectrogram_time.png": (f2, t2, s2, "sweep — classical STFT"),
        "spectrogram_quantum.png": (fq, tq, pq, "sweep — quantum shot histograms (4096/frame)"),
    }.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        db = 10 * np.log10(s + 1e-12)
        mesh = ax.pcolormesh(t, f, db, shading="auto", vmin=db.max() - 70, vmax=db.max())
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")
        fig.colorbar(mesh, ax=ax, label="dB")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)
        print(f"  wrote {fname}")

    _write_ablations_md()
    total = time.perf_counter() - t_start
    print(f"\ntotal render time: {total:.0f}s")
    return total


def _write_ablations_md():
    m = metrics
    lines = [
        "# qassette demo ablations",
        "",
        "All files in this directory are reproducible: `python scripts\\render_demo.py`",
        "(fixed seeds). Correlations are against the clean input chord.",
        "",
        "| ablation | files | metric | quantum principle |",
        "|---|---|---|---|",
        (f"| shots knob | shots_{{1,16,256,4096,65536}}.wav | corr "
         + " / ".join(f"{m[f'shots_{s}']:.3f}" for s in SHOT_LADDER)
         + " | **Measurement**: one shot collapses the frame to a single 8-bit sample; "
           "fidelity grows as sqrt(shots) because statistics, not the state, carry the song |"),
        (f"| readout | readout_z.wav vs readout_homodyne.wav | corr {m['readout_z']:.3f} vs "
         f"{m['readout_homodyne']:.3f} (z vs \\|x\\|: {m['readout_z_abs']:.3f}) "
         "| **Interference**: without the reference arm, Z readout gets only magnitudes — "
         "the rectified sound; homodyne interferes against a known state and recovers signs |"),
        (f"| QFT effects | fx_{{delay,chirp,ringmod,lowpass,fshift}}.wav | corr "
         + " / ".join(f"{m[f'fx_{s.split(chr(58))[0]}']:.3f}" for s in FX_SPECS)
         + " | **Superposition**: one 8-qubit state holds all 256 samples, so a handful of "
           "phase gates transforms the whole frame at once (low corr = the effect working) |"),
        (f"| memory + echo | memory_*.wav | corr at 500us: none {m['memory_0500us_none']:.3f}, "
         f"hahn {m['memory_0500us_hahn']:.3f}, cpmg {m['memory_0500us_cpmg']:.3f} "
         "| **Coherence**: storage dephases the qubits; echo pulses interfere the state with "
           "its own past and refocus slow noise |"),
        (f"| dephasing ablation | dephasing_{{coherent,memoryless}}_hahn.wav | corr "
         f"{m['dephasing_coherent_hahn']:.3f} vs {m['dephasing_memoryless_hahn']:.3f} "
         "| **Interference (in time)**: the echo only restores audio when the noise has "
           "memory; against memoryless dephasing the same pulse does nothing — this is why "
           "the noise model matters |"),
        ("| raw bits | bits_frame10.txt, spectrogram_{time,quantum}.png | — "
         "| **Measurement**: the literal 0/1 shot record is all that ever leaves the "
           "simulator; the spectrogram is built from those histograms alone |"),
        "",
        "WAV/PNG files are gitignored build artifacts; regenerate with the script.",
    ]
    (OUT / "ablations.md").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote ablations.md")


if __name__ == "__main__":
    main()
