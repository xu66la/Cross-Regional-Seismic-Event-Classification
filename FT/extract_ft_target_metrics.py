#!/usr/bin/env python3
"""从 FT 汇总 JSON 中提取 12 个迁移方向的目标域指标。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "FT"
    / "finetune_full_stft_r0.10_summary.json"
)

EXPECTED_DOMAINS = ("base", "enam", "msh", "hlp")
EXPECTED_SEEDS = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取 10% FT 的 12 个迁移方向目标域 Accuracy 和 Macro-F1"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_INPUT.parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("seeds") != EXPECTED_SEEDS:
        raise ValueError(f"期望 seeds={EXPECTED_SEEDS}，实际为 {data.get('seeds')}")
    if float(data.get("tgt_ratio", -1)) != 0.10:
        raise ValueError(f"期望 tgt_ratio=0.10，实际为 {data.get('tgt_ratio')}")

    expected_pairs = [
        (source, target)
        for source in EXPECTED_DOMAINS
        for target in EXPECTED_DOMAINS
        if source != target
    ]
    rows = []
    for source, target in expected_pairs:
        transfer = f"{source}->{target}"
        # 每个迁移方向下面有四个测试域；只取与 target 同名的目标域结果。
        metrics = data["summary"][transfer][target]
        rows.append(
            {
                "method": "FT",
                "transfer": transfer,
                "source": source,
                "target": target,
                "n": len(EXPECTED_SEEDS),
                "accuracy_mean": float(metrics["acc_mean"]),
                "accuracy_ci95": float(metrics["acc_CI95"]),
                "macro_f1_mean": float(metrics["macroF1_mean"]),
                "macro_f1_ci95": float(metrics["macroF1_CI95"]),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "ft_10pct_target_metrics_5seed.csv"
    json_path = args.out_dir / "ft_10pct_target_metrics_5seed.json"

    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "FT",
                "setting": "10% target-domain fine-tuning",
                "source_summary": str(args.input.resolve()),
                "seeds": EXPECTED_SEEDS,
                "num_transfers": len(rows),
                "rows": rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
