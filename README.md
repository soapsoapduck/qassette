# qassette

A quantum audio cassette: encode audio into quantum states, process it with
quantum circuits, and play it back. Qiskit + Aer, simulation only.

Built for the Qollab 48h quantum hackathon (Oct 9–11, 2026).

## Disclosure

The project scaffold (package layout, dependency pins, sample-WAV generator)
and the audio framing / amplitude-encoding round-trip pipeline were
prototyped by the author with AI assistance shortly before the event and
rebuilt at kickoff as the project's starting point, per the event rules on
disclosed starting material. Everything beyond that baseline is in-window
work.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts\make_samples.py
pytest
```

## CLI

```powershell
# WAV -> quantum tape (per-frame circuits, QPY + JSON header in one zip)
qassette encode samples\sine_440hz_2s.wav -o song.qtape

# measure every frame: homodyne keeps signs, z keeps magnitudes only
qassette measure song.qtape -o song.qbits --shots 4096 --readout homodyne

# play back from the measurement record alone
qassette decode song.qbits -o out.wav

# stare at the raw quantum shots (ancilla bit | data bits)
qassette bits song.qbits --frame 10 --max-shots 32

# all three steps + fidelity metrics; --keep-files keeps the intermediates
qassette roundtrip samples\sine_440hz_2s.wav -o out.wav --shots 4096 --readout homodyne --keep-files

# quantum effects: phase gates acting on the spectrum between QFT and IQFT
qassette roundtrip samples\a_major_chord_2s.wav -o fx.wav --shots 65536 --fx "delay:64,chirp:4,ringmod:16"

# non-unitary filters via post-selection (cutoffs in bins of sr/N Hz; optional
# softness angle < pi), and a single-sideband frequency shifter. One
# post-selected effect per chain; success rate is recorded in the .qbits header.
qassette roundtrip samples\a_major_chord_2s.wav -o lp.wav --shots 65536 --fx "lowpass:8"
qassette roundtrip samples\a_major_chord_2s.wav -o up.wav --shots 65536 --fx "fshift:8"

# store each frame in a simulated mechanical quantum memory before readout
# (T1/T2* from Bozkurt et al., Nat. Phys. 21, 1469 (2025)); echo pulses help
qassette roundtrip samples\sine_440hz_2s.wav -o mem.wav --shots 4096 --memory-time 300us --dd hahn

# measure in the frequency basis (right after the QFT) ...
qassette measure song.qtape -o song_freq.qbits --shots 4096 --readout z --basis freq

# ... or go straight to a quantum-vs-classical spectrogram image
qassette spectrogram song.qtape --shots 4096 -o spec.png
```

More shots = cleaner playback: RMS error falls as 1/sqrt(shots). Try 64
(buried in shot noise), 1024 (lo-fi tape hiss), 65536 (clean).
