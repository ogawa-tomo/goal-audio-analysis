"""Command-line interface -- orchestrates `clip/`, `groupstats.py`, and `compare.py`.

    goal-audio mark-onset <clip.wav> [--pick N | --time T]
    goal-audio analyze <clip1.wav> [clip2.wav ...] -o features.json
    goal-audio compare <features_a.json> <features_b.json> --label-a Premier --label-b LaLiga -o results/

Acquiring and cutting the source audio (before it's a clip ready to be
marked/analyzed) is out of scope for this CLI — see the README and
scripts/.

This is the one place in the package that knows about file-path
conventions (`<clip>.mark.json`, `<clip>.window.wav`): `clip/`'s modules
only ever receive paths as arguments, never derive or assume them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import compare
from .clip import mark, onset, window, spectral, formant, f0, zcr, hnr

SR = 22050
WINDOW_S = 2.0

DEFAULT_BAR_METRICS = [
    ("spectral_centroid_hz", "Spectral centroid (Hz)"),
    ("spectral_rolloff85_hz", "Rolloff 85% (Hz)"),
    ("spectral_bandwidth_hz", "Bandwidth (Hz)"),
    ("f0_median_hz", "F0 median (Hz)"),
    ("hnr_db", "HNR (dB)"),
    ("zero_crossing_rate", "Zero crossing rate"),
]


def parse_time(value: str) -> float:
    """Parse a time string as either plain seconds ("4.2") or "M:SS[.ms]" ("0:04.2")."""
    value = value.strip()
    if re.match(r"^\d+:\d+(\.\d+)?$", value):
        minutes_str, seconds_str = value.split(":")
        return int(minutes_str) * 60 + float(seconds_str)
    return float(value)


def _mark_path(clip_path: Path) -> Path:
    return clip_path.with_suffix(".mark.json")


def _window_path(clip_path: Path) -> Path:
    return clip_path.with_suffix(".window.wav")


def _prepare_window(clip_path: Path) -> Path:
    """Ensure `clip_path`'s onset is marked, then (re-)cut its analysis window.

    Raises `mark.MissingMarkError` if the clip has no onset mark yet
    (see `cmd_mark_onset`). The window file is always regenerated, so it
    can never go stale relative to whatever onset was most recently
    confirmed.
    """
    onset_time = mark.require(_mark_path(clip_path), "onset_marked_time_s")
    return window.extract_window(clip_path, onset_time, WINDOW_S, _window_path(clip_path), sr=SR)


def _analyze_one(clip_path: Path, with_formants: bool = True) -> dict:
    onset_time = mark.require(_mark_path(clip_path), "onset_marked_time_s")
    window_path = window.extract_window(clip_path, onset_time, WINDOW_S, _window_path(clip_path), sr=SR)

    spectral_feat = spectral.analyze(window_path, sr=SR)
    f0_feat = f0.analyze(window_path, sr=SR)
    zcr_val = zcr.analyze(window_path, sr=SR)

    f1_median = f2_median = hnr_val = None
    if with_formants:
        formant_feat = formant.analyze(window_path, sr=SR)
        f1_median, f2_median = formant_feat.f1_median_hz, formant_feat.f2_median_hz
        hnr_val = hnr.analyze(window_path, sr=SR)

    return {
        "file": clip_path.name,
        "onset_time_s": round(onset_time, 2),
        **spectral_feat.to_dict(),
        "zero_crossing_rate": zcr_val,
        **f0_feat.to_dict(),
        "f1_median_hz": f1_median,
        "f2_median_hz": f2_median,
        "hnr_db": hnr_val,
    }


def cmd_mark_onset(args):
    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"error: file not found: {clip_path}", file=sys.stderr)
        return 1

    if args.pick is not None and args.time is not None:
        print("error: --pick and --time can't both be given", file=sys.stderr)
        return 1

    mark_path = _mark_path(clip_path)
    existing = mark.load_mark(mark_path)

    if args.time is not None:
        try:
            selected = round(parse_time(args.time), 3)
        except ValueError:
            print(f"error: could not parse time: {args.time!r} (expected seconds or M:SS)", file=sys.stderr)
            return 1
        existing["onset_marked_time_s"] = selected
        mark.save_mark(mark_path, existing)
        print(f"saved: {mark_path} (onset_marked_time_s={selected}, free-form)")
        return 0

    candidates = onset.onset_candidates_for_clip(
        clip_path,
        window_s=args.window,
        height_ratio=args.height_ratio,
        min_separation_s=args.min_separation,
    )
    if not candidates:
        print(f"error: no candidates found for {clip_path.name}; try --time to specify manually", file=sys.stderr)
        return 1

    print(f"{clip_path.name}: {len(candidates)} candidate(s)")
    for i, t in enumerate(candidates, start=1):
        print(f"  [{i}] {t:.2f}s")

    if args.pick is None:
        print("\nlisten to each candidate, then pass --pick <number> (or --time <seconds>) to select one.")
        return 0

    if not (1 <= args.pick <= len(candidates)):
        print(f"error: --pick must be between 1 and {len(candidates)}", file=sys.stderr)
        return 1

    selected = candidates[args.pick - 1]
    existing["onset_marked_time_s"] = selected
    mark.save_mark(mark_path, existing)
    print(f"saved: {mark_path} (onset_marked_time_s={selected})")
    return 0


def cmd_analyze(args):
    results = []
    skipped = []
    for p in args.clips:
        try:
            results.append(_analyze_one(Path(p), with_formants=not args.no_formants))
        except mark.MissingMarkError as e:
            print(f"error: {e}", file=sys.stderr)
            skipped.append(p)
    if skipped:
        print(f"skipped {len(skipped)} unmarked clip(s); {len(results)} analyzed", file=sys.stderr)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"saved: {args.out}")
    else:
        print(text)


def cmd_compare(args):
    group_a = json.loads(Path(args.features_a).read_text(encoding="utf-8"))
    group_b = json.loads(Path(args.features_b).read_text(encoding="utf-8"))

    comparisons = compare.compare_groups(group_a, group_b, label_a=args.label_a, label_b=args.label_b)
    print(compare.format_table(comparisons, args.label_a, args.label_b))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        compare.plot_bar_comparison(
            group_a, group_b, DEFAULT_BAR_METRICS,
            out_dir / "comparison.png", args.label_a, args.label_b,
            title=f"{args.label_a} vs {args.label_b}",
        )
        if any(it.get("f1_median_hz") is not None for it in group_a + group_b):
            compare.plot_formant_chart(
                group_a, group_b, out_dir / "formant_chart.png", args.label_a, args.label_b,
            )
        print(f"plots saved under: {out_dir}")


def cmd_spectrum_plot(args):
    def load_curves(paths, curve_fn):
        curves = []
        skipped = []
        for p in paths:
            clip_path = Path(p)
            try:
                window_path = _prepare_window(clip_path)
            except mark.MissingMarkError as e:
                print(f"error: {e}", file=sys.stderr)
                skipped.append(p)
                continue
            curve = curve_fn(window_path, sr=SR)
            if curve is not None:
                curves.append(curve)
            else:
                skipped.append(p)
        if skipped:
            print(f"skipped {len(skipped)} clip(s) with no usable curve", file=sys.stderr)
        return curves

    out_dir = Path(args.out_dir)

    abs_a = load_curves(args.group_a, spectral.curve)
    abs_b = load_curves(args.group_b, spectral.curve)
    abs_out = out_dir / "spectrum_absolute.png"
    compare.plot_curve_comparison(
        abs_a, abs_b, abs_out, args.label_a, args.label_b,
        xlabel="Frequency (Hz)", ylabel="Normalized magnitude (shape, arbitrary units)",
        x_max=6000.0, normalize=True,
        title=f"Spectrum shape (post-onset window): {args.label_a} vs {args.label_b}",
    )
    print(f"saved: {abs_out}")

    fmt_a = load_curves(args.group_a, formant.curve)
    fmt_b = load_curves(args.group_b, formant.curve)
    fmt_out = out_dir / "formant_envelope.png"
    compare.plot_curve_comparison(
        fmt_a, fmt_b, fmt_out, args.label_a, args.label_b,
        xlabel="Frequency (Hz)", ylabel="Normalized LPC envelope (shape, arbitrary units)",
        x_max=5500.0, normalize=True,
        title=f"Formant (LPC) envelope shape: {args.label_a} vs {args.label_b}",
    )
    print(f"saved: {fmt_out}")

    hnr_a = load_curves(args.group_a, hnr.curve)
    hnr_b = load_curves(args.group_b, hnr.curve)
    hnr_out = out_dir / "hnr_over_time.png"
    compare.plot_curve_comparison(
        hnr_a, hnr_b, hnr_out, args.label_a, args.label_b,
        xlabel="Time since onset (s)", ylabel="HNR (dB)",
        x_max=None, normalize=False,
        title=f"HNR over time: {args.label_a} vs {args.label_b}",
    )
    print(f"saved: {hnr_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goal-audio")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "mark-onset",
        help="show onset-moment candidates for a clip, and mark the correct one",
    )
    p.add_argument("clip", help="path to the clip's WAV file")
    p.add_argument("--pick", type=int, default=None, help="candidate number to select (1-based)")
    p.add_argument("--time", type=str, default=None, help="if no candidate is correct, specify a time directly (seconds or M:SS)")
    p.add_argument("--height-ratio", type=float, default=0.4, dest="height_ratio")
    p.add_argument("--min-separation", type=float, default=1.5, dest="min_separation")
    p.add_argument("--window", type=float, default=8.0, help="how many seconds from the clip's start to search for candidates")
    p.set_defaults(func=cmd_mark_onset)

    p = sub.add_parser("analyze", help="extract acoustic features from one or more clips")
    p.add_argument("clips", nargs="+")
    p.add_argument("-o", "--out", help="write JSON here instead of stdout")
    p.add_argument("--no-formants", action="store_true", help="skip Praat-based analysis (formants F1/F2, HNR)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("compare", help="compare two groups of pre-computed features")
    p.add_argument("features_a", help="JSON file: list of feature dicts for group A")
    p.add_argument("features_b", help="JSON file: list of feature dicts for group B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("-o", "--out-dir", help="directory to save comparison plots into")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("spectrum-plot", help="overlay each group's post-onset spectrum/envelope/HNR shape")
    p.add_argument("--group-a", nargs="+", required=True, dest="group_a", help="clip wav paths for group A")
    p.add_argument("--group-b", nargs="+", required=True, dest="group_b", help="clip wav paths for group B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("-o", "--out-dir", default="results", help="directory to save the plot into")
    p.set_defaults(func=cmd_spectrum_plot)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
