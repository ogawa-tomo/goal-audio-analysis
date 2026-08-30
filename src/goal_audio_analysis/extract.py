"""Audio acquisition: download a source video's audio and cut short clips from it.

Depends on the external `yt-dlp` and `ffmpeg` executables being on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def download_audio(url: str, out_path: str | Path, audio_format: str = "wav") -> Path:
    """Download only the audio track of `url` and save it as `out_path`.

    `out_path`'s extension is ignored; the final file will have the
    `audio_format` extension (default wav).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", audio_format,
            "-o", f"{stem}.%(ext)s",
            url,
        ],
        check=True,
    )
    result = stem.with_suffix(f".{audio_format}")
    if not result.exists():
        raise FileNotFoundError(f"expected download output not found: {result}")
    return result


def extract_clip(
    src_wav: str | Path,
    out_wav: str | Path,
    center_s: float,
    lead_s: float = 1.0,
    duration_s: float = 7.0,
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
    """Raise RuntimeError if yt-dlp or ffmpeg are not on PATH."""
    missing = [tool for tool in ("yt-dlp", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"required tool(s) not found on PATH: {', '.join(missing)}")
