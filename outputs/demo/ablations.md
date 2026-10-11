# qassette demo ablations

All files in this directory are reproducible: `python scripts\render_demo.py`
(fixed seeds). Correlations are against the clean input chord.

| ablation | files | metric | quantum principle |
|---|---|---|---|
| shots knob | shots_{1,16,256,4096,65536}.wav | corr 0.067 / 0.284 / 0.755 / 0.977 / 0.999 | **Measurement**: one shot collapses the frame to a single 8-bit sample; fidelity grows as sqrt(shots) because statistics, not the state, carry the song |
| readout | readout_z.wav vs readout_homodyne.wav | corr -0.001 vs 0.978 (z vs \|x\|: 0.985) | **Interference**: without the reference arm, Z readout gets only magnitudes — the rectified sound; homodyne interferes against a known state and recovers signs |
| QFT effects | fx_{delay,chirp,ringmod,lowpass,fshift}.wav | corr -0.119 / -0.282 / -0.007 / 0.613 / 0.001 | **Superposition**: one 8-qubit state holds all 256 samples, so a handful of phase gates transforms the whole frame at once (low corr = the effect working) |
| memory + echo | memory_*.wav | corr at 500us: none 0.016, hahn 0.415, cpmg 0.841 | **Coherence**: storage dephases the qubits; echo pulses interfere the state with its own past and refocus slow noise |
| dephasing ablation | dephasing_{coherent,memoryless}_hahn.wav | corr 0.979 vs 0.009 | **Interference (in time)**: the echo only restores audio when the noise has memory; against memoryless dephasing the same pulse does nothing — this is why the noise model matters |
| raw bits | bits_frame10.txt, spectrogram_{time,quantum}.png | — | **Measurement**: the literal 0/1 shot record is all that ever leaves the simulator; the spectrogram is built from those histograms alone |

WAV/PNG files are gitignored build artifacts; regenerate with the script.