"""Clip extraction: cut a short window out of a local audio file.

How the source audio file itself is acquired is out of scope here — see the
README for guidance (e.g. loopback-recording playback rather than
downloading, when the source is a service like YouTube whose terms of
service prohibit unauthorized downloads).

Depends on the external `ffmpeg` executable being on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_clip(
    src_wav: str | Path,
    out_wav: str | Path,
    center_s: float,
    lead_s: float = 3.0,
    duration_s: float = 9.0,
) -> Path:
    """Cut a `duration_s`-long clip from `src_wav`, starting `lead_s` seconds
    before `center_s` (e.g. before a goal timestamp).
    """
    src_wav = Path(src_wav)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    start = max(0.0, center_s - lead_s)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_wav),
            "-ss", str(start),
            "-t", str(duration_s),
            str(out_wav),
        ],
        check=True,
    )
    return out_wav


def check_tools_available() -> None:
    """Raise RuntimeError if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("required tool not found on PATH: ffmpeg")
