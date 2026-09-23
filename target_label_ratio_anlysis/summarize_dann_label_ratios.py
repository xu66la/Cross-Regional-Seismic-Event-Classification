#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""汇总 DANN 在不同目标域标注比例下的目标域测试指标。

输出 5 个标注比例 x 12 个有向迁移方向，共 60 行。每行汇总
seed 0--4 的 target-test Accuracy 和 Macro-F1。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


DOMAINS = ("base", "enam", "msh", "hlp")
SEEDS = tuple(range(5))
RATIOS = (0.00, 0.01, 0.05, 0.10, 0.20)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "DANN_target_label_ratio"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 DANN 0%/1%/5%/10%/20% 标注比例的目标域 Accuracy 和 Macro-F1"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许使用当前已有 seed 生成阶段性结果；最终结果请勿使用此选项",
    )
    return parser.parse_args()


def result_path(root: Path, source: str, target: str, seed: int, ratio: float) -> Path:
    run_name = (
        f"stft_dann_seed{seed}"
        if math.isclose(ratio, 0.0, abs_tol=1e-12)
        else f"stft_ssda_seed{seed}_r{ratio:.2f}"
    )
    return (
        root
        / f"{source}_to_{target}"
        / run_name
        / "results.json"
    )


def load_result(
    path: Path,
    source: str,
    target: str,
    seed: int,
    ratio: float,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    actual_meta = (
        str(result.get("source")),
        str(result.get("target")),
        int(result.get("seed", -1)),
    )
    expected_meta = (source, target, seed)
    if actual_meta != expected_meta:
        raise ValueError(f"元数据不匹配：期望 {expected_meta}，实际 {actual_meta}")

    actual_ratio = float(result.get("hyper_params", {}).get("tgt_label_ratio", -1))
    if not math.isclose(actual_ratio, ratio, abs_tol=1e-9):
        raise ValueError(f"标注比例不匹配：期望 {ratio:.2f}，实际 {actual_ratio}")

    target_test = result["final"]["tgt_test"]
    accuracy = float(target_test["accuracy"])
    macro_f1 = float(target_test["macro_f1"])
    if not (math.isfinite(accuracy) and math.isfinite(macro_f1)):
        raise ValueError("accuracy 或 macro_f1 不是有限数值")

    return {
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "result_json": str(path),
    }


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "ci95": None}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    missing_paths: list[Path] = []

    for ratio in RATIOS:
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue

                runs = []
                missing_seeds = []
                for seed in SEEDS:
                    path = result_path(root, source, target, seed, ratio)
                    if not path.is_file():
                        missing_paths.append(path)
                        missing_seeds.append(seed)
                        continue
                    try:
                        runs.append(load_result(path, source, target, seed, ratio))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise RuntimeError(f"无法读取有效结果 {path}: {exc}") from exc

                accuracy = stats([float(run["accuracy"]) for run in runs])
                macro_f1 = stats([float(run["macro_f1"]) for run in runs])
                rows.append(
                    {
                        "label_ratio": ratio,
                        "label_percent": int(round(ratio * 100)),
                        "transfer": f"{source}->{target}",
                        "source": source,
                        "target": target,
                        "n": len(runs),
                        "missing_seeds": missing_seeds,
                        "accuracy_mean": accuracy["mean"],
                        "accuracy_std": accuracy["std"],
                        "accuracy_ci95": accuracy["ci95"],
                        "macro_f1_mean": macro_f1["mean"],
                        "macro_f1_std": macro_f1["std"],
                        "macro_f1_ci95": macro_f1["ci95"],
                        "runs": runs,
                    }
                )

    if missing_paths and not args.allow_incomplete:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:20])
        remainder = len(missing_paths) - min(20, len(missing_paths))
        extra = f"\n  ... 另外还有 {remainder} 个缺失文件" if remainder else ""
        raise SystemExit(
            f"结果尚未完整：缺少 {len(missing_paths)} 个 results.json。\n"
            f"{preview}{extra}\n"
            "训练全部完成后重新运行，或添加 --allow-incomplete 查看阶段性结果。"
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dann_label_ratio_target_metrics_5seed.csv"
    json_path = out_dir / "dann_label_ratio_target_metrics_5seed.json"

    csv_fields = [
        "label_ratio",
        "label_percent",
        "transfer",
        "source",
        "target",
        "n",
        "missing_seeds",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_ci95",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_ci95",
    ]
    metric_fields = [
        "accuracy_mean",
        "accuracy_std",
        "accuracy_ci95",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_ci95",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row[field] for field in csv_fields}
            csv_row["label_ratio"] = f"{float(row['label_ratio']):.2f}"
            csv_row["missing_seeds"] = ",".join(map(str, row["missing_seeds"]))
            for field in metric_fields:
                value = csv_row[field]
                csv_row[field] = "" if value is None else f"{float(value):.6f}"
            writer.writerow(csv_row)

    payload = {
        "method": "DANN",
        "setting": "DANN UDA/SSDA target-label-ratio sweep",
        "ratios": list(RATIOS),
        "expected_seeds": list(SEEDS),
        "expected_rows": len(RATIOS) * len(DOMAINS) * (len(DOMAINS) - 1),
        "complete": not missing_paths,
        "missing_result_count": len(missing_paths),
        "source_root": str(root),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    if missing_paths:
        print(f"完整性：阶段性结果，缺少 {len(missing_paths)} 个文件")
    else:
        print("完整性：完整（5 个比例 × 12 个方向 × 5 个 seed）")


if __name__ == "__main__":
    main()
