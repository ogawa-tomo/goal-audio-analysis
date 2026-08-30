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
]


def cmd_extract_clip(args):
    out = extract.extract_clip(args.src, args.out, center_s=args.center, lead_s=args.lead, duration_s=args.duration)
    print(f"saved: {out}")


def cmd_analyze(args):
    results = [
        features.analyze_clip(
            p, with_formants=not args.no_formants, peak_search_window_s=args.peak_search_window,
            smooth_window_s=args.smooth_window,
        ).to_dict()
        for p in args.clips
    ]
    for r in results:
        diff = r.get("peak_vs_mark_diff_s")
        if diff is not None and abs(diff) > args.mark_threshold:
            print(
                f"WARNING: {r['file']}: peak_time_s ({r['peak_time_s']}) differs from the "
                f"human mark ({r['human_marked_time_s']}) by {diff:+.2f}s (threshold "
                f"{args.mark_threshold}s). The detected peak may not be the intended event -- "
                f"inspect this clip (see scripts/mark_goal_moment.py).",
                file=sys.stderr,
            )
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
        help="seconds before `center` to start the clip (default 3.0; pair with "
             "analyze's --peak-search-window so that lead + search-window's "
             "post-`center` reach stays consistent -- see README)",
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
    p.add_argument(
        "--peak-search-window", type=float, default=6.0, dest="peak_search_window",
        help="only search the first N seconds of the clip for the RMS peak (default 6.0). "
             "Should match extract-clip's --lead so the search window's far end stays "
             "`center + (search_window - lead)` seconds after the goal timestamp -- see README",
    )
    p.add_argument(
        "--smooth-window", type=float, default=0.3, dest="smooth_window",
        help="moving-average width (seconds) applied to the RMS envelope before peak-searching "
             "(default 0.3). Prevents a brief single-frame click (e.g. a recording glitch) from "
             "outscoring a genuine multi-second crowd swell. Set to 0 to disable",
    )
    p.add_argument(
        "--mark-threshold", type=float, default=2.0, dest="mark_threshold",
        help="if a clip has a human mark (see scripts/mark_goal_moment.py) and it differs from "
             "the detected peak_time_s by more than this many seconds (default 2.0), print a "
             "warning -- the detected peak may be the wrong event",
    )
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
