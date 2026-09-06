"""Fundamental frequency (pitch) analysis of an already-windowed clip (see `window.py`)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import librosa


@dataclass
class F0Features:
    f0_median_hz: Optional[float]
    f0_mean_hz: Optional[float]
    voiced_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(window_wav_path: str | Path, sr: int = 22050) -> F0Features:
    """Estimate F0 (pitch) via pYIN.

    Crowd noise is broadband and often non-periodic, so pYIN's voiced/
    unvoiced decision frequently comes back mostly unvoiced --
    `voiced_fraction` reports how much of the window that happened for,
    so a median/mean computed from very few voiced frames can be
    recognized as an outlier-prone estimate rather than trusted at face
    value.
    """
    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)

    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
    )
    f0_voiced = f0[~np.isnan(f0)]

    return F0Features(
        f0_median_hz=round(float(np.median(f0_voiced)), 1) if len(f0_voiced) else None,
        f0_mean_hz=round(float(np.mean(f0_voiced)), 1) if len(f0_voiced) else None,
        voiced_fraction=round(float(np.mean(~np.isnan(f0))), 3),
    )
