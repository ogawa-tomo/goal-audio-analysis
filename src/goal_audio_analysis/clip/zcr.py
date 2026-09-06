"""Zero-crossing-rate analysis of an already-windowed clip (see `window.py`).

A single scalar (no per-frequency-bin form), so unlike spectral/formant/
hnr there's no `curve()` here -- just this one number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa


def analyze(window_wav_path: str | Path, sr: int = 22050) -> float:
    """Return the mean zero-crossing rate over the whole of `window_wav_path`.

    Higher values indicate more noise-like/high-frequency content.
    """
    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    return round(float(np.mean(zcr)), 5)
