#!/usr/bin/env python3
"""将 DAPO 风格 JSON/JSONL（含 prompt 对话列表 + label）转为 VERL RLHF parquet。

VERL 期望字段见 verl/utils/dataset/README.md：需要 data_source、prompt、reward_model.ground_truth 等。
原始数据示例：
  {"prompt": [{"role": "user", "content": "..."}], "label": "95"}

用法:
  python prepare_dapo_style_parquet.py --input train.jsonl --output train.parquet
  python prepare_dapo_style_parquet.py --input test.jsonl --output test.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


def _iter_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        else:
            yield data
    else:
        raise ValueError(f"Unsupported input suffix: {path.suffix} (use .jsonl or .json)")


def _to_verl_row(obj: dict, data_source: str) -> dict:
    if "prompt" not in obj:
        raise KeyError("Each record must contain 'prompt' (chat messages list).")
    label = obj.get("label", obj.get("answer", obj.get("ground_truth")))
    if label is None:
        raise KeyError("Each record needs 'label' (or 'answer' / 'ground_truth').")
    # math_dapo.compute_score 使用 str；与官方示例中 list 形式相比，字符串更直接
    gt = label if isinstance(label, str) else str(label)
    return {
        "prompt": obj["prompt"],
        "data_source": data_source,
        "ability": obj.get("ability", "math"),
        "reward_model": {"style": "rule", "ground_truth": gt},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="JSONL 或 JSON 数组文件")
    p.add_argument("--output", type=Path, required=True, help="输出 .parquet 路径")
    p.add_argument(
        "--data_source",
        type=str,
        default="math_dapo",
        help="写入 data_source，对应 verl.utils.reward_score.default_compute_score 中的 math_dapo 分支",
    )
    args = p.parse_args()

    rows = [_to_verl_row(o, args.data_source) for o in _iter_records(args.input)]
    if not rows:
        raise SystemExit("No records found in input.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
