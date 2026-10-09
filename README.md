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
python scripts\make_samples.py
pytest
```
