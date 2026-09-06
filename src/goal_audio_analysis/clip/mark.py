"""Read/write the JSON sidecar file used to persist human decisions about a clip.

Generic: doesn't know or care what keys the dict contains (e.g.
`onset_marked_time_s`) -- callers decide the key names, this only handles
the file I/O. Keeping that knowledge out of this module is what lets
`window.py`/`clip/spectral.py` etc. stay unaware that a sidecar file (or
any particular key name) exists at all; only the orchestration layer
(`cli.py`) needs to know the convention.
"""
from __future__ import annotations

import json
from pathlib import Path


class MissingMarkError(ValueError):
    """Raised by `require()` when a mark file is missing the requested key."""


def load_mark(mark_path: str | Path) -> dict:
    """Return the dict stored at `mark_path`, or `{}` if it doesn't exist or isn't valid JSON."""
    mark_path = Path(mark_path)
    if not mark_path.exists():
        return {}
    try:
        return json.loads(mark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_mark(mark_path: str | Path, data: dict) -> None:
    """Write `data` to `mark_path` as indented JSON."""
    Path(mark_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def require(mark_path: str | Path, key: str) -> float:
    """Return `float(data[key])` from the mark at `mark_path`, or raise `MissingMarkError`."""
    data = load_mark(mark_path)
    if key not in data:
        raise MissingMarkError(f"{Path(mark_path).name} has no {key!r}.")
    return float(data[key])
