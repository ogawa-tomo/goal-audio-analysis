r"""機械的に検出したゴール瞬間の候補から、正しいものを人間が選ぶ。

`goal_audio_analysis.features.analyze_clip`と同じ候補検出ロジックでこのクリップの
候補時刻を列挙し、番号付きで表示する。それぞれの候補の時刻を、通常のメディアプレイヤーや
`tools/audio_time_reader.html` などで実際に聴いて確認したうえで、正しいものの番号を
`--pick` で選ぶと、その時刻をサイドカーファイル(`<clip>.mark.json`)に保存する。以降
`goal-audio analyze` を実行すると、このファイルを自動的に読み込み、選ばれた時刻を実際の
特徴量計算(attack/decay時間・スペクトル特徴量など)の基準点として使う。

**重要**: `--peak-search-window` は、実際に`goal-audio analyze`に渡す値と必ず一致させる
こと。ループバック録音したクリップ(`data/clips/*_lb.wav`)は`8`を使うこと -- この既定値
(6.0)は`extract-clip`の`--lead 3 --duration 9`との組み合わせ用で、ループバック録音の
`--duration 14`とは合わない。窓が狭すぎると、本来検出されるべき候補が探索範囲外に
切り捨てられ、見つからなくなる。

## 使い方

    python scripts/mark_goal_moment.py <clip.wav> --peak-search-window 8
        候補一覧を表示するだけ(まだ選択しない)
    python scripts/mark_goal_moment.py <clip.wav> --peak-search-window 8 --pick <番号>
        候補の中から正しいものを選び、選択結果を保存する

## 設計メモ

以前のバージョンは、人間が自由に時刻を入力する方式だった。しかしこの方式には、
音声を一度だけ聴いて「ゴールだと感じた瞬間」に反応してその数値をそのまま申告する、
という実際の運用では、反応速度のズレが計測値にそのまま混入するという問題がある
(何度も聞き直せば理論上ゼロにできるが、実際にそうする保証はない)。

これに対し、「機械が検出した候補の中から正しいものを選ぶ」方式であれば、候補同士は
通常2秒以上離れているため、多少の反応の遅れがあっても「どの候補を選んだか」という
判断そのものは揺らがない。かつ、実際に解析へ渡される時刻は人間の申告値ではなく、
機械が算出した精密な値になるため、反応速度のズレが計測精度に影響しない。

なお、この方式が機能するには「正しい候補がそもそも一覧に存在する」ことが前提になる。
持続的な盛り上がり(単発の鋭いピークがない)を専用に検出する仕組みも試したが、実際の
17クリップで検証したところ、探索窓を正しく広げてさえいれば、振幅ピーク型の検出だけで
持続区間内の小さな起伏を拾えており、追加の検出ロジックは不要と判明したため削除した
(むしろ大半のクリップで、単なる音量減衰の揺らぎを別候補として誤検出していた)。それでも
候補に正解が存在しない場合は、検出ロジック自体(閾値やこのスクリプトの候補検出
パラメータ)を見直す必要がある。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from goal_audio_analysis import features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="対象のWAVファイルのパス")
    parser.add_argument("--pick", type=int, default=None, help="正しい候補の番号(1始まり)。省略すると候補一覧のみ表示する")
    parser.add_argument(
        "--peak-search-window", type=float, default=6.0, dest="peak_search_window",
        help="`goal-audio analyze`に渡すのと同じ値を指定すること(既定6.0)",
    )
    parser.add_argument("--peak-height-ratio", type=float, default=0.8, dest="peak_height_ratio")
    parser.add_argument("--peak-min-separation", type=float, default=2.0, dest="peak_min_separation")
    args = parser.parse_args(argv)

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"error: file not found: {clip_path}", file=sys.stderr)
        return 1

    result = features.analyze_clip(
        clip_path,
        with_formants=False,
        peak_search_window_s=args.peak_search_window,
        candidate_height_ratio=args.peak_height_ratio,
        candidate_min_separation_s=args.peak_min_separation,
    )
    candidates = result.candidate_peak_times_s

    print(f"{clip_path.name}: {len(candidates)} candidate(s)")
    for i, t in enumerate(candidates, start=1):
        marker = " (loudest)" if abs(t - result.default_peak_time_s) < 1e-6 else ""
        print(f"  [{i}] {t:.2f}s{marker}")

    if args.pick is None:
        print("\n各候補の時刻を実際に聴いて確認し、--pick <番号> で選択してください。")
        return 0

    if not (1 <= args.pick <= len(candidates)):
        print(f"error: --pick must be between 1 and {len(candidates)}", file=sys.stderr)
        return 1

    selected = candidates[args.pick - 1]
    mark_path = clip_path.with_suffix(".mark.json")
    mark_path.write_text(
        json.dumps({"selected_peak_time_s": selected}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved: {mark_path} (selected_peak_time_s={selected})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
