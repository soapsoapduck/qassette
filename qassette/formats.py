""".qbits tape format: raw quantum measurement outcomes plus a JSON header.

A .qbits file is an uncompressed NumPy .npz archive with two entries:

- "header": a JSON string with keys
    version, sr, N, hop, length, mode ("z" | "homodyne"), basis ("time"),
    shots, norms (per-frame L2 norms), effects (applied effect chain)
- "outcomes": uint16 array of shape (num_frames, shots); each value is a
    Qiskit little-endian integer outcome — data index in the low n bits,
    homodyne ancilla in bit n (leftmost character of the printed bitstring).

The file is self-contained: decode needs nothing but this file.
"""

import io
import json
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy

FORMAT_VERSION = 1
QTAPE_VERSION = 1
_SHOT_MODES = ("z", "homodyne")


@dataclass
class Qbits:
    header: dict
    outcomes: np.ndarray  # (num_frames, shots) uint16

    @property
    def n(self) -> int:
        return int(self.header["N"]).bit_length() - 1


def _expected_frames(N: int, hop: int, length: int) -> int:
    # mirror frame_signal: pre-pad N - hop, last frame covers the final sample
    return ((N - hop) + length - 1) // hop + 1


def save_qbits(
    path: str | Path,
    outcomes: np.ndarray,
    *,
    sr: int,
    N: int,
    hop: int,
    length: int,
    mode: str,
    shots: int,
    norms: list[float],
    effects: list[str] | tuple = (),
    basis: str = "time",
) -> None:
    """Write a .qbits tape. outcomes: (num_frames, shots) integer array."""
    if mode not in _SHOT_MODES:
        raise ValueError(f"mode must be one of {_SHOT_MODES}, got {mode!r}")
    if basis not in ("time", "freq"):
        raise ValueError(f"basis must be 'time' or 'freq', got {basis!r}")
    outcomes = np.asarray(outcomes)
    if outcomes.ndim != 2 or outcomes.shape != (len(norms), shots):
        raise ValueError(
            f"outcomes shape {outcomes.shape} does not match (len(norms), shots) = ({len(norms)}, {shots})"
        )
    bits = (int(N).bit_length() - 1) + (1 if mode == "homodyne" else 0)
    if bits > 16:
        raise ValueError(
            f".qbits packs outcomes as uint16: N={N} in {mode} mode needs {bits}-bit "
            f"outcomes; the format caps N at 2**15 for homodyne / 2**16 for z"
        )
    if outcomes.size and (int(outcomes.min()) < 0 or int(outcomes.max()) >= 2**bits):
        raise ValueError(
            f"outcome values span [{int(outcomes.min())}, {int(outcomes.max())}], outside "
            f"the {bits}-bit range implied by N={N}, mode={mode}"
        )
    packed = outcomes.astype(np.uint16)
    header = {
        "version": FORMAT_VERSION,
        "sr": int(sr),
        "N": int(N),
        "hop": int(hop),
        "length": int(length),
        "mode": mode,
        "basis": basis,
        "shots": int(shots),
        "norms": [float(v) for v in norms],
        "effects": list(effects),
        "crc32": zlib.crc32(packed.tobytes()),
    }
    with open(path, "wb") as f:
        np.savez(f, header=json.dumps(header), outcomes=packed)


def load_qbits(path: str | Path) -> Qbits:
    """Load and validate a .qbits tape; raises ValueError on inconsistency."""
    with np.load(path, allow_pickle=False) as npz:
        try:
            header = json.loads(npz["header"].item())
            outcomes = npz["outcomes"]
        except KeyError as e:
            raise ValueError(f"{path}: not a .qbits file (missing entry {e})") from e
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported .qbits version {header.get('version')!r}")
    mode, N, hop = header["mode"], header["N"], header["hop"]
    if mode not in _SHOT_MODES:
        raise ValueError(f"{path}: unknown readout mode {mode!r}")
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"{path}: header N={N} is not a power of two")
    if not 0 < hop <= N:
        raise ValueError(f"{path}: header hop={hop} invalid for N={N}")
    if outcomes.shape != (len(header["norms"]), header["shots"]):
        raise ValueError(
            f"{path}: outcomes shape {outcomes.shape} does not match header "
            f"(frames={len(header['norms'])}, shots={header['shots']})"
        )
    expected = _expected_frames(N, hop, header["length"])
    if outcomes.shape[0] != expected:
        raise ValueError(
            f"{path}: header says N={N}, hop={hop}, length={header['length']} "
            f"=> {expected} frames, but file holds {outcomes.shape[0]}"
        )
    bits = (N.bit_length() - 1) + (1 if mode == "homodyne" else 0)
    if outcomes.size and int(outcomes.max()) >= 2**bits:
        raise ValueError(
            f"{path}: outcome value {int(outcomes.max())} does not fit in {bits} bits "
            f"implied by header N={N}, mode={mode} — header is corrupt"
        )
    if "crc32" in header and zlib.crc32(np.ascontiguousarray(outcomes).tobytes()) != header["crc32"]:
        raise ValueError(f"{path}: outcomes corrupted (CRC mismatch)")
    return Qbits(header=header, outcomes=outcomes)


@dataclass
class Qtape:
    header: dict
    circuits: list[QuantumCircuit]


def save_qtape(
    path: str | Path,
    circuits: list[QuantumCircuit],
    *,
    sr: int,
    N: int,
    hop: int,
    length: int,
    norms: list[float],
    effects: list[str] | tuple = (),
) -> None:
    """Write a .qtape: per-frame circuits (QPY) + JSON header in one zip."""
    if len(circuits) != len(norms):
        raise ValueError(f"{len(circuits)} circuits but {len(norms)} norms")
    header = {
        "version": QTAPE_VERSION,
        "sr": int(sr),
        "N": int(N),
        "hop": int(hop),
        "length": int(length),
        "frames": len(circuits),
        "norms": [float(v) for v in norms],
        "effects": list(effects),
    }
    buf = io.BytesIO()
    qpy.dump(list(circuits), buf)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("header.json", json.dumps(header))
        z.writestr("circuits.qpy", buf.getvalue())


def load_qtape(path: str | Path) -> Qtape:
    """Load and validate a .qtape; raises ValueError on inconsistency."""
    try:
        with zipfile.ZipFile(path) as z:
            header = json.loads(z.read("header.json"))
            circuits = qpy.load(io.BytesIO(z.read("circuits.qpy")))
    except (zipfile.BadZipFile, KeyError) as e:
        raise ValueError(f"{path}: not a .qtape file ({e})") from e
    if header.get("version") != QTAPE_VERSION:
        raise ValueError(f"{path}: unsupported .qtape version {header.get('version')!r}")
    N, hop = header["N"], header["hop"]
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"{path}: header N={N} is not a power of two")
    if len(circuits) != header["frames"] or len(circuits) != len(header["norms"]):
        raise ValueError(
            f"{path}: {len(circuits)} circuits vs header frames={header['frames']}, "
            f"norms={len(header['norms'])}"
        )
    if len(circuits) != _expected_frames(N, hop, header["length"]):
        raise ValueError(
            f"{path}: header N/hop/length imply {_expected_frames(N, hop, header['length'])} "
            f"frames, file holds {len(circuits)}"
        )
    return Qtape(header=header, circuits=circuits)


def to_ascii(qbits: Qbits, frame: int, max_shots: int = 16) -> str:
    """Render one frame's shots as literal bitstrings, ancilla separated.

    Homodyne lines look like "a | dddddddd" where a is the ancilla (qubit n,
    leftmost) and d are the n data bits (qubit n-1 .. 0). Z lines are just
    the n data bits.
    """
    header = qbits.header
    n = qbits.n
    shown = qbits.outcomes[frame][:max_shots]
    homodyne = header["mode"] == "homodyne"
    width = n + 1 if homodyne else n
    lines = [
        f"frame {frame} | mode={header['mode']} | showing {len(shown)}/{header['shots']} shots",
        ("a | " + "d" * n) if homodyne else "d" * n,
    ]
    for o in shown:
        bits = format(int(o), f"0{width}b")
        lines.append(f"{bits[0]} | {bits[1:]}" if homodyne else bits)
    return "\n".join(lines)
