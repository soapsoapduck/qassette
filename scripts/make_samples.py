"""Generate 16 kHz mono test WAVs in samples/: sine, A-major chord, and a sine sweep."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp

SAMPLE_RATE = 16_000
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return (peak * x / np.max(np.abs(x))).astype(np.float32)


def make_sine(duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return _normalize(np.sin(2 * np.pi * freq * t))


def make_a_major_chord(duration: float = 2.0) -> np.ndarray:
    # A4, C#5, E5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    chord = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 554.37, 659.25))
    return _normalize(chord)


def make_sweep(duration: float = 3.0, f0: float = 100.0, f1: float = 4000.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return _normalize(chirp(t, f0=f0, f1=f1, t1=duration, method="logarithmic"))


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sine_440hz_2s.wav": make_sine(),
        "a_major_chord_2s.wav": make_a_major_chord(),
        "sweep_100_4000hz_3s.wav": make_sweep(),
    }
    for name, data in outputs.items():
        path = SAMPLES_DIR / name
        sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")
        print(f"wrote {path} ({len(data) / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
