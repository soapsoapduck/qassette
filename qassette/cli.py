"""qassette CLI: record audio onto quantum tape and play it back."""

import time
from pathlib import Path

import numpy as np
import typer

from .audio_io import frame_signal, load_audio, save_audio
from .decode import decode_qbits
from .encode import encode_frame
from .formats import load_qbits, load_qtape, save_qbits, save_qtape, to_ascii
from .readout import sample_homodyne, sample_z

app = typer.Typer(
    help="qassette: encode audio into quantum circuits, measure, and play back. Simulation only.",
    no_args_is_help=True,
    add_completion=False,
)

READOUTS = ("homodyne", "z")


def _check_readout(readout: str) -> str:
    if readout not in READOUTS:
        raise typer.BadParameter(f"readout must be one of {READOUTS}")
    return readout


@app.command()
def encode(
    in_wav: Path = typer.Argument(..., exists=True, dir_okay=False, help="input WAV"),
    output: Path = typer.Option(..., "--output", "-o", help="output .qtape"),
    n: int = typer.Option(256, "--frame-size", "-N", help="frame size (power of two)"),
    hop: int = typer.Option(128, help="hop size"),
    sr: int = typer.Option(16000, help="sample rate to load at"),
) -> None:
    """Encode a WAV into a .qtape of per-frame quantum circuits."""
    t0 = time.perf_counter()
    x = load_audio(in_wav, sr)
    frames = frame_signal(x, N=n, hop=hop)
    circuits, norms = [], []
    with typer.progressbar(frames, label=f"encode  {len(frames)} frames") as bar:
        for f in bar:
            qc, nm = encode_frame(f)
            circuits.append(qc)
            norms.append(nm)
    save_qtape(output, circuits, sr=sr, N=n, hop=hop, length=len(x), norms=norms)
    typer.echo(
        f"encode  {in_wav.name} -> {output} "
        f"({output.stat().st_size:,} bytes) in {time.perf_counter() - t0:.1f}s"
    )


@app.command()
def measure(
    qtape: Path = typer.Argument(..., exists=True, dir_okay=False, help="input .qtape"),
    output: Path = typer.Option(..., "--output", "-o", help="output .qbits"),
    shots: int = typer.Option(4096, help="shots per frame"),
    readout: str = typer.Option("homodyne", help="homodyne (signed) or z (magnitudes)"),
    seed: int = typer.Option(None, help="base RNG seed (frame i uses seed+i)"),
) -> None:
    """Measure every frame of a .qtape into a .qbits file of raw shot outcomes."""
    _check_readout(readout)
    t0 = time.perf_counter()
    tape = load_qtape(qtape)
    sampler = sample_homodyne if readout == "homodyne" else sample_z
    rows = []
    with typer.progressbar(
        list(enumerate(tape.circuits)), label=f"measure {len(tape.circuits)} frames"
    ) as bar:
        for i, qc in bar:
            rows.append(sampler(qc, shots, seed=None if seed is None else seed + i))
    h = tape.header
    save_qbits(
        output, np.stack(rows), sr=h["sr"], N=h["N"], hop=h["hop"], length=h["length"],
        mode=readout, shots=shots, norms=h["norms"], effects=h["effects"],
    )
    typer.echo(
        f"measure {qtape.name} -> {output} ({readout}, {shots} shots/frame, "
        f"{output.stat().st_size:,} bytes) in {time.perf_counter() - t0:.1f}s"
    )


@app.command()
def decode(
    qbits: Path = typer.Argument(..., exists=True, dir_okay=False, help="input .qbits"),
    output: Path = typer.Option(..., "--output", "-o", help="output WAV"),
    renormalize: bool = typer.Option(
        False, help="clamp frame estimates to unit norm (tames low-shot clipping)"
    ),
) -> None:
    """Reconstruct a WAV from a .qbits file alone."""
    t0 = time.perf_counter()
    sr = load_qbits(qbits).header["sr"]
    x = decode_qbits(qbits, renormalize=renormalize)
    save_audio(output, np.clip(x, -1.0, 1.0), sr)
    typer.echo(f"decode  {qbits.name} -> {output} in {time.perf_counter() - t0:.1f}s")


@app.command()
def bits(
    qbits: Path = typer.Argument(..., exists=True, dir_okay=False, help="input .qbits"),
    frame: int = typer.Option(0, help="frame index to display"),
    max_shots: int = typer.Option(32, help="shots to display"),
) -> None:
    """Print a frame's raw shots as literal 0/1 bitstrings."""
    typer.echo(to_ascii(load_qbits(qbits), frame=frame, max_shots=max_shots))


@app.command()
def roundtrip(
    in_wav: Path = typer.Argument(..., exists=True, dir_okay=False, help="input WAV"),
    output: Path = typer.Option(..., "--output", "-o", help="output WAV"),
    shots: int = typer.Option(4096, help="shots per frame"),
    readout: str = typer.Option("homodyne", help="homodyne (signed) or z (magnitudes)"),
    keep_files: bool = typer.Option(False, "--keep-files", help="keep .qtape/.qbits"),
    seed: int = typer.Option(None, help="base RNG seed"),
    renormalize: bool = typer.Option(False, help="clamp frame estimates to unit norm"),
) -> None:
    """encode -> measure -> decode in one go, with fidelity metrics."""
    _check_readout(readout)
    qtape_path = output.with_suffix(".qtape")
    qbits_path = output.with_suffix(".qbits")
    t0 = time.perf_counter()
    encode(in_wav=in_wav, output=qtape_path, n=256, hop=128, sr=16000)
    measure(qtape=qtape_path, output=qbits_path, shots=shots, readout=readout, seed=seed)
    decode(qbits=qbits_path, output=output, renormalize=renormalize)
    x = load_audio(in_wav, 16000)
    y = load_audio(output, 16000)
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    corr = float(np.corrcoef(y, x)[0, 1])
    typer.echo(
        f"roundtrip done in {time.perf_counter() - t0:.1f}s | "
        f"RMS error {rmse:.4f} | correlation {corr:.4f}"
    )
    if not keep_files:
        qtape_path.unlink(missing_ok=True)
        qbits_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
