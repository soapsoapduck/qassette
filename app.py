"""qassette — Streamlit front-end. One file; all physics lives in the package.

Run: .venv\\Scripts\\streamlit.exe run app.py
Measurement uses the operator-cached analytic sampling path from
qassette.memory (exact at t=0, tested to 1e-12 against the circuit path),
so the controls are live rather than minutes-per-click.
"""

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from qassette.audio_io import frame_signal, load_audio
from qassette.decode import frames_to_audio
from qassette.effects import build_fx_circuit, chain_names, parse_fx
from qassette.encode import encode_frame
from qassette.formats import Qbits, save_qbits, save_qtape, to_ascii
from qassette.memory import MemoryParams, TapeEvaluator, memory_distribution
from qassette.readout import estimate_homodyne, estimate_z, success_rate
from qassette.viz import spectrogram_comparison

ROOT = Path(__file__).resolve().parent
SR, N, HOP, NQ = 16000, 256, 128, 8
MAX_SECONDS = 10.0
SEED = 1234
SHOT_STEPS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]

st.set_page_config(page_title="qassette", page_icon="📼", layout="wide")


def wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(np.clip(y, -1, 1), dtype=np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def load_input(src_bytes: bytes | None, sample_name: str | None) -> np.ndarray:
    src = io.BytesIO(src_bytes) if src_bytes is not None else ROOT / "samples" / f"{sample_name}.wav"
    x = load_audio(src, SR)
    return x[: int(MAX_SECONDS * SR)]


@st.cache_data(show_spinner="encoding frames into quantum states…")
def encode_cached(x: np.ndarray, spec: str):
    """Frames -> circuits -> post-fx state vectors. Cached so measurement
    knobs (shots/readout/basis/memory) never trigger a re-encode."""
    chain = parse_fx(spec, NQ)
    fx = build_fx_circuit(NQ, chain)
    ev = TapeEvaluator()
    circuits, norms, phis = [], [], []
    for f in frame_signal(x, N=N, hop=HOP):
        qc, nm = encode_frame(f, fx=fx)
        circuits.append(qc)
        norms.append(nm)
        phis.append(ev.output_vector(qc))
    return circuits, norms, np.stack(phis), fx.num_qubits > NQ, chain_names(chain)


@st.cache_data(show_spinner=False)
def gate_counts(spec: str) -> dict:
    from qiskit import transpile

    chain = parse_fx(spec, NQ)
    fx = build_fx_circuit(NQ, chain)
    qc, _ = encode_frame(np.sin(np.arange(N) / 7.0), fx=fx)
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    ops = dict(t.count_ops())
    return {"qubits": qc.num_qubits, "cx": int(ops.get("cx", 0)), "total": int(sum(ops.values()))}


def bitstream_image(rows: np.ndarray, width_bits: int, max_shots: int = 160) -> np.ndarray:
    """Shots as pixels: one row per shot, one column per bit (MSB left)."""
    shown = rows[:max_shots]
    bits = (shown[:, None] >> np.arange(width_bits - 1, -1, -1)[None, :]) & 1
    img = (255 - bits * 255).astype(np.uint8)
    return np.kron(img, np.ones((3, 6), dtype=np.uint8))  # upscale for visibility


# ================================================================ sidebar
with st.sidebar:
    st.header("Input")
    source = st.radio("source", ["sample", "upload"], horizontal=True, label_visibility="collapsed")
    upload_bytes, sample_name = None, None
    if source == "sample":
        sample_name = st.selectbox("sample", [p.stem for p in sorted((ROOT / "samples").glob("*.wav"))])
    else:
        up = st.file_uploader(f"WAV (trimmed to {MAX_SECONDS:.0f} s)", type=["wav", "flac", "ogg"])
        upload_bytes = up.getvalue() if up is not None else None

    st.header("Effects chain")
    fx_pick = st.multiselect("effects", ["delay", "chirp", "ringmod", "lowpass", "highpass", "fshift"],
                             help="applied in order, between QFT and inverse QFT; one filter max")
    tokens = []
    for name in fx_pick:
        if name == "delay":
            tokens.append(f"delay:{st.slider('delay [samples]', 1, 255, 64)}")
        elif name == "chirp":
            tokens.append(f"chirp:{st.slider('chirp [dispersion]', 0.5, 8.0, 4.0, 0.5):g}")
        elif name == "ringmod":
            tokens.append(f"ringmod:{st.slider('ringmod [bins of 62.5 Hz]', 1, 64, 16)}")
        elif name in ("lowpass", "highpass"):
            cut = st.slider(f"{name} cutoff [bins]", 1, 127, 8 if name == "lowpass" else 4)
            soft = st.slider(f"{name} softness [rad]", 0.5, float(np.pi), float(np.pi), 0.1)
            tokens.append(f"{name}:{cut}" if abs(soft - np.pi) < 0.05 else f"{name}:{cut}:{soft:g}")
        elif name == "fshift":
            tokens.append(f"fshift:{st.slider('fshift [bins]', 1, 63, 8)}")
    spec = ",".join(tokens)

    st.header("Memory")
    mem_us = st.slider("storage time [µs]", 0, 2000, 0, 50)
    dd = st.radio("dynamical decoupling", ["none", "hahn", "cpmg"], horizontal=True)

    st.header("Measurement")
    shots = st.select_slider("shots per frame (log)", SHOT_STEPS, value=4096)
    readout = st.radio("readout", ["homodyne", "z"], horizontal=True)
    basis = st.radio("basis", ["time", "freq"], horizontal=True)

    st.header("Output")
    output_mode = st.radio("output mode", ["Audio", "Quantum bits", "Both"])

# ================================================================= main
tab_deck, tab_proof = st.tabs(["📼 tape deck", "🔬 is it quantum?"])

with tab_deck:
    st.title("📼 qassette")
    st.caption("audio → amplitude-encoded qubits → gates on the spectrum → "
               "mechanical quantum memory → shots → audio. Simulation only.")
    if upload_bytes is None and sample_name is None:
        st.info("Upload a WAV in the sidebar to begin.")
        st.stop()
    if basis == "freq" and readout != "z":
        st.error("basis='freq' requires readout='z' — frequency-bin magnitudes carry no homodyne signs.")
        st.stop()
    try:
        x = load_input(upload_bytes, sample_name)
    except Exception as e:  # unreadable upload
        st.error(f"could not read that file: {e}")
        st.stop()
    if len(x) < N:
        st.error(f"need at least {N} samples ({N / SR * 1000:.0f} ms) of audio.")
        st.stop()
    if not np.any(np.abs(x) > 1e-5):
        st.warning("input is nearly silent — the tape will faithfully store silence.")

    try:
        circuits, norms, phis, filtered, fx_names = encode_cached(x, spec)
    except ValueError as e:
        st.error(f"effect chain: {e}")
        st.stop()

    t0 = time.perf_counter()
    mem = MemoryParams(t=mem_us * 1e-6, dd=dd)
    rows, amps = [], []
    fb = NQ if filtered else None
    prog = st.progress(0.0, text="measuring…")
    for i in range(len(phis)):
        rng = np.random.default_rng(SEED + i)
        probs = memory_distribution(phis[i], NQ, mem, rng, readout, basis)
        counts = rng.multinomial(shots, probs)
        outs = np.repeat(np.arange(len(probs), dtype=np.uint16), counts)
        rng.shuffle(outs)
        rows.append(outs)
        amps.append(estimate_homodyne(outs, N, fb) if readout == "homodyne" else estimate_z(outs, N, fb))
        if i % 16 == 0 or i == len(phis) - 1:
            prog.progress((i + 1) / len(phis), text=f"measuring… frame {i + 1}/{len(phis)}")
    prog.empty()
    rows = np.stack(rows)
    rates = [success_rate(r, NQ, homodyne=(readout == "homodyne")) for r in rows] if filtered else None
    y = frames_to_audio(amps, norms, hop=HOP, length=len(x)) if basis == "time" else None
    runtime = time.perf_counter() - t0

    qb = Qbits(
        header={"version": 1, "sr": SR, "N": N, "hop": HOP, "length": len(x), "mode": readout,
                "basis": basis, "shots": shots, "norms": list(norms), "effects": fx_names,
                "filter": filtered, "success_rates": rates,
                "memory": None if mem_us == 0 else mem.header_dict()},
        outcomes=rows,
    )

    # info panel
    g = gate_counts(spec)
    cols = st.columns(6)
    cols[0].metric("qubits / frame", g["qubits"] + (1 if readout == "homodyne" else 0),
                   help="data + filter ancilla (+ homodyne ancilla at readout)")
    cols[1].metric("CNOTs / frame", f"{g['cx']:,}")
    cols[2].metric("shots / frame", f"{shots:,}")
    cols[3].metric("frames", len(phis))
    cols[4].metric("post-selection", "—" if rates is None else f"{np.mean(rates):.0%}")
    cols[5].metric("runtime", f"{runtime:.1f} s")

    if output_mode in ("Audio", "Both"):
        a, b = st.columns(2)
        a.subheader("original")
        a.audio(wav_bytes(x))
        b.subheader("off the quantum tape")
        if y is not None:
            denom = float(np.std(y)) * float(np.std(x))
            corr = float(np.corrcoef(y, x)[0, 1]) if denom > 1e-12 else 0.0
            b.audio(wav_bytes(y))
            b.caption(f"correlation with original: {corr:.3f}")
        else:
            b.info("freq-basis tapes hold only spectra — see the spectrogram below.")

    if output_mode in ("Quantum bits", "Both"):
        st.subheader("the raw quantum bits")
        frame_ix = st.number_input("frame", 0, len(phis) - 1, min(10, len(phis) - 1))
        width_bits = NQ + (1 if filtered else 0) + (1 if readout == "homodyne" else 0)
        c1, c2 = st.columns([2, 1])
        c1.image(bitstream_image(rows[frame_ix], width_bits, max_shots=64), width=300,
                 caption=f"frame {frame_ix}: first {min(64, shots)} shots, one row per shot "
                         f"(black = 1). Columns: {'homodyne | ' if readout == 'homodyne' else ''}"
                         f"{'filter | ' if filtered else ''}8 data bits.")
        c2.code(to_ascii(qb, frame=int(frame_ix), max_shots=16))
        d1, d2, d3 = st.columns(3)
        if y is not None:
            d1.download_button("⬇ decoded .wav", wav_bytes(y), "qassette_out.wav")
        qb_buf = io.BytesIO()
        save_qbits(qb_buf, rows, sr=SR, N=N, hop=HOP, length=len(x), mode=readout, shots=shots,
                   norms=list(norms), effects=fx_names, basis=basis, filter_ancilla=filtered,
                   success_rates=rates, memory=None if mem_us == 0 else mem.header_dict())
        d2.download_button("⬇ raw shots .qbits", qb_buf.getvalue(), "qassette_out.qbits")
        qt_buf = io.BytesIO()
        save_qtape(qt_buf, circuits, sr=SR, N=N, hop=HOP, length=len(x), norms=list(norms),
                   effects=fx_names, filter_ancilla=filtered)
        d3.download_button("⬇ circuits .qtape", qt_buf.getvalue(), "qassette_out.qtape")

    with st.expander("spectrograms — classical vs quantum", expanded=(basis == "freq")):
        if basis == "freq":
            fqb = qb
        else:
            frows = []
            for i in range(len(phis)):
                rng = np.random.default_rng(SEED + 99_000 + i)
                probs = memory_distribution(phis[i], NQ, mem, rng, "z", "freq")
                counts = rng.multinomial(max(shots, 2048), probs)
                frows.append(np.repeat(np.arange(len(probs), dtype=np.uint16), counts))
            fqb = Qbits(
                header={"version": 1, "sr": SR, "N": N, "hop": HOP, "length": len(x), "mode": "z",
                        "basis": "freq", "shots": max(shots, 2048), "norms": list(norms),
                        "effects": fx_names, "filter": filtered},
                outcomes=np.stack(frows),
            )
        st.pyplot(spectrogram_comparison(fqb, x, SR))

with tab_proof:
    st.title("🔬 is it quantum?")
    md = ROOT / "outputs" / "demo" / "ablations.md"
    if md.exists():
        st.markdown(md.read_text(encoding="utf-8"))
        demo = ROOT / "outputs" / "demo"
        groups = {
            "the shots knob (measurement statistics)": [f"shots_{s}.wav" for s in (1, 256, 65536)],
            "readout ablation (interference)": ["readout_z.wav", "readout_homodyne.wav"],
            "memory + echo (coherence)": ["memory_0200us_none.wav", "memory_0200us_hahn.wav",
                                          "memory_0500us_cpmg.wav"],
            "echo needs noise with memory": ["dephasing_coherent_hahn.wav", "dephasing_memoryless_hahn.wav"],
        }
        for title, files in groups.items():
            st.subheader(title)
            cols = st.columns(len(files))
            for c, fname in zip(cols, files):
                p = demo / fname
                if p.exists():
                    c.caption(fname)
                    c.audio(p.read_bytes())
    else:
        st.info("run `python scripts\\render_demo.py` to generate the ablation evidence.")
