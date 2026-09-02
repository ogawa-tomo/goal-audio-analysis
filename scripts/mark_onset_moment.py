r"""歓声が盛り上がり始めた瞬間(立ち上がりの瞬間)を人間が選び、記録する。

`goal_audio_analysis.onset`が、前後1秒ずつのスペクトル形状の変化量(コサイン距離)を
0.05秒刻みで計算し、局所的な極大点を候補として列挙する。それぞれの候補の時刻を、
メディアプレイヤーなどで実際に聴いて確認したうえで、正しいものの番号を`--pick`で選ぶと、
その時刻をサイドカーファイル(`<clip>.mark.json`)に`onset_marked_time_s`として保存する。
以降`goal-audio analyze`を実行すると、この値がそのまま解析の基準点として使われる。

候補がどれも正しくない場合のために、`--time`で自由に時刻を指定することもできる(この
プロジェクトの17クリップでの検証では、候補が0.6秒以内に正解を含まないケースはなかったが、
新しいクリップで同じ保証はないため、逃げ道として残している)。

## 使い方

    python scripts/mark_onset_moment.py <clip.wav>
        候補一覧を表示するだけ(まだ選択しない)
    python scripts/mark_onset_moment.py <clip.wav> --pick <番号>
        候補の中から正しいものを選び、選択結果を保存する
    python scripts/mark_onset_moment.py <clip.wav> --time <秒数 または 分:秒>
        候補のどれも正しくない場合、自由に時刻を指定して保存する

## 設計メモ

このプロジェクトでは、`scripts/mark_goal_moment.py`(音量ピークを自由入力でマークする、
今は使われていない旧方式)を経て、一度は「候補から選ぶ」方式(音量ピーク候補ベース)を試し、
「正解がそもそも候補に含まれないケースがある」という問題に直面して自由入力方式に戻した、
という経緯がある。今回の立ち上がり検出は、それとは別のスペクトル形状変化という手法であり、
17クリップ全件で候補が正解の近く(0.6秒以内)に見つかることを検証済みのため、「候補提示+
人間が選ぶ」を基本方針としつつ、`--time`による自由入力を保険として残す形にしている。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from goal_audio_analysis import onset


def parse_time(value: str) -> float:
    """Parse a time string as either plain seconds ("4.2") or "M:SS[.ms]" ("0:04.2")."""
    value = value.strip()
    if re.match(r"^\d+:\d+(\.\d+)?$", value):
        minutes_str, seconds_str = value.split(":")
        return int(minutes_str) * 60 + float(seconds_str)
    return float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="対象のWAVファイルのパス")
    parser.add_argument("--pick", type=int, default=None, help="正しい候補の番号(1始まり)")
    parser.add_argument("--time", type=str, default=None, help="候補にない場合、自由に時刻を指定(秒数 または 分:秒)")
    parser.add_argument("--height-ratio", type=float, default=0.4, dest="height_ratio")
    parser.add_argument("--min-separation", type=float, default=1.5, dest="min_separation")
    parser.add_argument("--window", type=float, default=8.0, help="候補を探す範囲(クリップ先頭から何秒か)")
    args = parser.parse_args(argv)

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"error: file not found: {clip_path}", file=sys.stderr)
        return 1

    if args.pick is not None and args.time is not None:
        print("error: --pick と --time は同時に指定できません", file=sys.stderr)
        return 1

    mark_path = clip_path.with_suffix(".mark.json")
    existing = {}
    if mark_path.exists():
        try:
            existing = json.loads(mark_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    if args.time is not None:
        try:
            selected = round(parse_time(args.time), 3)
        except ValueError:
            print(f"error: could not parse time: {args.time!r} (expected seconds or M:SS)", file=sys.stderr)
            return 1
        existing["onset_marked_time_s"] = selected
        mark_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
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
        print("\n各候補の時刻を実際に聴いて確認し、--pick <番号> (または --time <秒数>) で選択してください。")
        return 0

    if not (1 <= args.pick <= len(candidates)):
        print(f"error: --pick must be between 1 and {len(candidates)}", file=sys.stderr)
        return 1

    selected = candidates[args.pick - 1]
    existing["onset_marked_time_s"] = selected
    mark_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {mark_path} (onset_marked_time_s={selected})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
