r"""人間が知覚したゴールの瞬間を記録する。

これは、`goal-audio analyze`によるピーク検出(音量ベース・アルゴリズム的)とは独立に、
「人間が実際に聞いて、ゴールだと感じた瞬間」を記録するための補助スクリプト。両者を
突き合わせることで、ピーク検出が別の場面(スルーパスへの歓声、演出音など)を誤って
捉えていないかを検証できる。

## 使い方

1. 対象のWAVファイルを、**通常のメディアプレイヤー**(Windowsメディアプレイヤー、VLC等、
   何でもよい)で開いて再生する
2. ゴールだと感じた瞬間に一時停止し、プレイヤーの画面に表示されている再生位置を読み取る
3. 以下のようにこのスクリプトを実行し、読み取った時刻を渡す

    python scripts/mark_goal_moment.py <clip.wav> <time>

`<time>` は秒数(例: `4.2`)、または `分:秒` 形式(例: `0:04.2`)のどちらでも指定できる。

これで、`<clip>`と同じ場所に`<clip>.mark.json`が保存される。以降`goal-audio analyze`を
実行すると、このファイルを自動的に読み込み、アルゴリズムのピーク検出結果と比較する。

## 設計メモ

以前のバージョンは、スクリプト自身がスピーカーで再生してキー入力を検知する方式だったが、
「`play()`を呼んでから実際に音が聞こえるまでの遅延」を測定・補正する必要が生じ、
その補正自体の精度も安定しなかった。市販のメディアプレイヤーは再生位置の表示を
正確に保つように作られているため、**そこに表示された時刻をそのまま読み取ってもらう**
方が、実装が単純な上に信頼できる。これは音声データを扱う準備工程の一部として、
意図的にgoal_audio_analysisライブラリ/CLIには含めていない。検出結果との突き合わせ
(数値比較)はライブラリ側(`features.analyze_clip`が自動で行う)で行う。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


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
    parser.add_argument("time", help="ゴールだと感じた再生位置。秒数(例: 4.2)または 分:秒(例: 0:04.2)")
    args = parser.parse_args(argv)

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"error: file not found: {clip_path}", file=sys.stderr)
        return 1

    try:
        human_marked_time_s = round(parse_time(args.time), 3)
    except ValueError:
        print(f"error: could not parse time: {args.time!r} (expected seconds or M:SS)", file=sys.stderr)
        return 1

    mark_path = clip_path.with_suffix(".mark.json")
    mark_path.write_text(
        json.dumps({"human_marked_time_s": human_marked_time_s}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved: {mark_path} (human_marked_time_s={human_marked_time_s})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
