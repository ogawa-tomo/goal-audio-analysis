# goal-audio-analysis

ゴール時の歓声など、群衆の音声データを解析するためのツール。

- **`features.py`**: 1クリップの音声から音響特徴量(音量ピーク・立ち上がり時間・スペクトル重心・
  ロールオフ・帯域幅・F0・フォルマントF1/F2)を抽出する
- **`compare.py`**: 特徴量(dictのリスト)を2群受け取り、Welchのt検定で比較する。音声処理には
  依存しないので、音声以外の2群比較にも流用できる
- **`extract.py`**: `yt-dlp`/`ffmpeg`を使って音声のダウンロードとクリップ切り出しを行う
- **`plotting.py`**: 比較結果の棒グラフ・母音(F1-F2)図を生成する
- **`cli.py`**: 上記をコマンドラインから呼び出せるようにするエントリポイント

## セットアップ

```bash
pip install -e .
```

`ffmpeg`実行ファイルもPATH上に必要(音声抽出に使用)。

## 使い方

```bash
# 1. 音声取得
goal-audio extract-audio "https://youtu.be/xxxx" data/raw/source.wav

# 2. ゴールタイムスタンプ(秒)を中心にクリップを切り出す
goal-audio extract-clip data/raw/source.wav data/clips/goal1.wav 74

# 3. 特徴量抽出(複数クリップまとめて)
goal-audio analyze data/clips/premier_*.wav -o results/premier_features.json
goal-audio analyze data/clips/laliga_*.wav -o results/laliga_features.json

# 4. 2群比較(表を標準出力 + グラフをresults/以下に保存)
goal-audio compare results/premier_features.json results/laliga_features.json \
  --label-a Premier --label-b LaLiga -o results/
```

Pythonから直接呼ぶ場合:

```python
from goal_audio_analysis import features, compare, plotting

f = features.analyze_clip("data/clips/goal1.wav")

group_a = [features.analyze_clip(p).to_dict() for p in premier_clips]
group_b = [features.analyze_clip(p).to_dict() for p in laliga_clips]
comparisons = compare.compare_groups(group_a, group_b, label_a="Premier", label_b="LaLiga")
print(compare.format_table(comparisons))
```

## ディレクトリ構成

```
.
├── src/goal_audio_analysis/   # ライブラリ本体
├── data/
│   ├── raw/                    # ダウンロードした元音声(Git管理外)
│   └── clips/                  # 切り出したクリップ(Git管理外)
├── results/                    # 解析結果(json/png、Git管理対象)
└── reports/                    # 調査記録・レポート
```

`data/`以下の音声ファイルは`.gitignore`で除外している。フォルダ構造だけ`.gitkeep`で保持している。

## 既知の注意点

- ピーク検出はクリップ先頭4秒以内に制限している(既定値)。それより後にスタジアムの演出音
  (ジングル・サイレン等)が入ると誤検出することがあったため
- フォルマント(F1/F2)解析はPraat(Burg法)を使用しているが、本来1人の声道を前提とした手法。
  群衆音への適用は「集団としてのスペクトル包絡のピーク」の近似であり、厳密な音声学的
  フォルマントではないことに留意
- F0(pYIN)は群衆のブロードバンドな歓声には本質的に不安定。有声フレーム率(`voiced_fraction`)
  が低いクリップのF0値は参考程度に扱うこと

詳しい調査記録は [`reports/`](reports/) を参照。
