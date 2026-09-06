"""Spectral shape analysis of an already-windowed clip (see `window.py`)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import librosa


@dataclass
class SpectralFeatures:
    spectral_centroid_hz: float
    spectral_rolloff85_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(window_wav_path: str | Path, sr: int = 22050) -> SpectralFeatures:
    """Compute scalar spectral features over the whole of `window_wav_path`."""
    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    return SpectralFeatures(
        spectral_centroid_hz=round(float(np.mean(centroid)), 1),
        spectral_rolloff85_hz=round(float(np.mean(rolloff)), 1),
        spectral_bandwidth_hz=round(float(np.mean(bandwidth)), 1),
        spectral_flatness=round(float(np.mean(flatness)), 5),
    )


def curve(window_wav_path: str | Path, sr: int = 22050) -> tuple[np.ndarray, np.ndarray]:
    """Return `(freqs, spectrum)` -- the raw average magnitude spectrum shape.

    Same window `analyze()` computes its scalar summary from, but
    returned as the actual per-bin shape instead of a centroid/rolloff/
    bandwidth/flatness reduction.
    """
    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return freqs, S.mean(axis=1)
