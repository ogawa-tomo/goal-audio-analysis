"""Command-line interface.

    goal-audio extract-clip <src.wav> <out.wav> <center_seconds>
    goal-audio analyze <clip1.wav> [clip2.wav ...] -o features.json
    goal-audio compare <features_a.json> <features_b.json> --label-a Premier --label-b LaLiga -o results/

How to acquire the source audio file itself (`src.wav` above) is out of
scope for this CLI — see the README.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import extract, features, compare, plotting

DEFAULT_BAR_METRICS = [
    ("attack_time_s", "Attack time (s)"),
    ("spectral_centroid_hz", "Spectral centroid (Hz)"),
    ("spectral_rolloff85_hz", "Rolloff 85% (Hz)"),
    ("spectral_bandwidth_hz", "Bandwidth (Hz)"),
    ("f0_median_hz", "F0 median (Hz)"),
    ("increase_centroid_hz", "Increase centroid (Hz)"),
    ("increase_rolloff85_hz", "Increase rolloff 85% (Hz)"),
    ("zero_crossing_rate_delta", "ZCR delta (post-pre)"),
]


def cmd_extract_clip(args):
    out = extract.extract_clip(args.src, args.out, center_s=args.center, lead_s=args.lead, duration_s=args.duration)
    print(f"saved: {out}")


def cmd_analyze(args):
    results = []
    skipped = []
    for p in args.clips:
        try:
            results.append(features.analyze_clip(p, with_formants=not args.no_formants).to_dict())
        except features.MissingMarkError as e:
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
        plotting.plot_bar_comparison(
            group_a, group_b, DEFAULT_BAR_METRICS,
            out_dir / "comparison.png", args.label_a, args.label_b,
            title=f"{args.label_a} vs {args.label_b}",
        )
        if any(it.get("f1_median_hz") is not None for it in group_a + group_b):
            plotting.plot_formant_chart(
                group_a, group_b, out_dir / "formant_chart.png", args.label_a, args.label_b,
            )
        print(f"plots saved under: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goal-audio")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-clip", help="cut a short clip from a source wav")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("center", type=float, help="timestamp (seconds) to center the clip on")
    p.add_argument(
        "--lead", type=float, default=3.0,
        help="seconds before `center` to start the clip (default 3.0) -- err generously here, "
             "since the goal moment itself is later specified exactly via "
             "scripts/mark_goal_moment.py, not detected automatically",
    )
    p.add_argument(
        "--duration", type=float, default=9.0,
        help="total clip length in seconds (default 9.0)",
    )
    p.set_defaults(func=cmd_extract_clip)

    p = sub.add_parser("analyze", help="extract acoustic features from one or more clips")
    p.add_argument("clips", nargs="+")
    p.add_argument("-o", "--out", help="write JSON here instead of stdout")
    p.add_argument("--no-formants", action="store_true", help="skip formant (F1/F2) analysis")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("compare", help="compare two groups of pre-computed features")
    p.add_argument("features_a", help="JSON file: list of feature dicts for group A")
    p.add_argument("features_b", help="JSON file: list of feature dicts for group B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("-o", "--out-dir", help="directory to save comparison plots into")
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
