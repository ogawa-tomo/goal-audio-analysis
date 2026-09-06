"""Harmonics-to-noise ratio (HNR, tonal-vs-noisy) analysis of an already-windowed clip (see `window.py`)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def analyze(window_wav_path: str | Path, sr: int = 22050) -> Optional[float]:
    """Estimate median HNR (dB) via Praat's cross-correlation method.

    HNR measures how periodic/tonal a sound is versus how noise-like: a
    high value means a strong, stable periodic component dominates (e.g.
    a crowd chanting roughly in unison), a low or negative value means
    broadband noise dominates (e.g. an unstructured roar with no shared
    pitch). This treats the crowd as if it were one periodic source,
    which is only an approximation -- but unlike F0, it doesn't require
    pitch-tracking to succeed on a frame to produce a value, so it stays
    defined even where `f0.F0Features.voiced_fraction` is low.

    The cross-correlation method (`to_harmonicity_cc`) is used rather
    than the autocorrelation one, per Praat's own guidance that cc is
    more robust on noisy/non-speech signals. Undefined frames come back
    from parselmouth as the sentinel value -200.0 and are excluded before
    taking the median. Returns `None` if every frame is undefined.
    """
    import parselmouth
    import librosa

    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    harmonicity = snd.to_harmonicity_cc()
    values = harmonicity.values.flatten()
    values = values[values != -200.0]
    if len(values) == 0:
        return None
    return round(float(np.median(values)), 2)


def curve(
    window_wav_path: str | Path,
    sr: int = 22050,
    time_step: float = 0.01,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return `(times, hnr_db)` -- HNR over time, frame by frame.

    `analyze()`'s return value is the median of this same curve.
    Returned frame-by-frame here instead, since HNR -- unlike the
    spectrum/envelope curves -- isn't defined per frequency: it's a
    per-time-frame measure of periodicity, so time is the only axis that
    makes sense for it. `times` is seconds since the start of
    `window_wav_path`. Undefined frames (Praat's -200 sentinel) come
    back as `nan` here rather than being dropped, so gaps in periodicity
    stay visible instead of silently vanishing.

    Returns `None` if the window is too short to yield any frame.
    """
    import parselmouth
    import librosa

    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    harmonicity = snd.to_harmonicity_cc(time_step=time_step)
    times = harmonicity.xs()
    if len(times) == 0:
        return None
    values = harmonicity.values.flatten()
    values = np.where(values == -200.0, np.nan, values)
    return times, values
