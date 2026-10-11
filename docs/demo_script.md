# qassette — 2-minute demo video shot list

All audio from `outputs/demo/` (regenerate: `python scripts\render_demo.py`).
Screen: the Streamlit app (`streamlit run app.py`) unless noted. Record audio
clean — the noise in the clips is the content.

| time | shot | audio | say roughly |
|---|---|---|---|
| 0:00–0:12 | **Hook.** App title card; play the two clips back to back | `memory_0200us_none.wav` → `memory_0200us_hahn.wav` | "This is a song stored on quantum tape — destroyed by 200 microseconds of decoherence. And this is the *same tape*, rescued by one echo pulse. qassette is a cassette deck where the tape is a quantum state." |
| 0:12–0:30 | **The bits.** App → output mode "Quantum bits": bit-stream image + ASCII panel; zoom on one line | none | "Every frame of audio becomes 8 qubits. All that ever comes back is this — raw measurement bits, one 8-bit sample per shot. The song exists only in the statistics." |
| 0:30–0:48 | **Shots knob.** Slider 1 → 256 → 65536, playing each | `shots_1.wav`, `shots_256.wav`, `shots_65536.wav` | "One shot per frame is pure static. 256 shots, the chord fights through. 65 thousand, it's clean. The hiss is not an effect — it's shot noise, falling exactly as one over root S." |
| 0:48–1:10 | **Effects + interference.** Add ringmod in the app (show CNOT counter jump); then the readout A/B | `fx_ringmod.wav`, then `readout_z.wav` → `readout_homodyne.wav` | "Effects are gates on the spectrum — this ring mod is a phase adder on the frequency register. And amplitudes have signs: measure naively and you only get magnitudes — this rectified buzz. Interfere against a reference — homodyne — and the signed waveform comes back." |
| 1:10–1:40 | **Memory + echo.** Memory slider 0 → 500 µs, dd none → hahn → cpmg, play the ladder; cut to the calibration figure | `memory_0500us_none.wav`, `memory_0500us_hahn.wav`, `memory_0500us_cpmg.wav` | "The tape ages like a real device: our noise is calibrated to a 2025 Nature Physics mechanical qubit memory — T2-star 64 microseconds. Free decay kills the music. A Hahn echo refocuses the slow noise; CPMG holds even longer. And the control: against memoryless noise the echo does nothing — refocusing only works because the noise has memory." (optionally flash `dephasing_memoryless_hahn.wav`) |
| 1:40–2:00 | **Close.** Spectrogram pair on screen; end on the repo/Qollab page | `shots_65536.wav` under voice | "Simulation only, every claim ablated and reproducible with fixed seeds — the repo plays itself. A 2-second song costs 33 megabytes of quantum measurements. Terrible storage. Wonderful physics. qassette." |

Tips: keep each audio clip ≥2 s before talking over it; show the correlation
metric changing in the app when the knobs move; the CNOTs/post-selection
tiles are good close-ups.
