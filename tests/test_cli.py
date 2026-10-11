from pathlib import Path

import numpy as np
import pytest

from qassette.audio_io import load_audio
from qassette.cli import decode as cli_decode
from qassette.cli import encode as cli_encode
from qassette.cli import measure as cli_measure

SR = 16000


@pytest.mark.parametrize("fx,readout", [("", "homodyne"), ("lowpass:12", "homodyne"), ("", "z")])
def test_cli_measure_fast_path_round_trips(tmp_path, fx, readout):
    """The no-memory measure path samples analytically (t=0 fast path);
    the decoded audio must still round-trip faithfully."""
    wav_in = Path(__file__).resolve().parent.parent / "samples" / "sine_440hz_2s.wav"
    qt, qb, out = tmp_path / "t.qtape", tmp_path / "t.qbits", tmp_path / "t.wav"
    cli_encode(in_wav=wav_in, output=qt, n=256, hop=128, sr=SR, fx=fx)
    cli_measure(qtape=qt, output=qb, shots=4096, readout=readout, basis="time",
                seed=7, memory_time="0", dd="none", trajectories=8)
    cli_decode(qbits=qb, output=out, renormalize=False)
    x = load_audio(wav_in)
    y = load_audio(out)
    if readout == "homodyne":
        assert np.corrcoef(y, x)[0, 1] > 0.9  # 440 Hz passes a bin-12 lowpass
    else:
        assert np.corrcoef(y, np.abs(x))[0, 1] > 0.9  # z keeps magnitudes
