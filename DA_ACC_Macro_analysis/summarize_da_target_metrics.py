"""汇总 DAN、DSAN、DANN 和 FT 的 10% 目标域测试指标。

对三种领域适应方法的 12 个有向迁移方向，读取 seed 0--4 的
results.json；FT 则从其 5-seed 汇总 JSON 中只取对应目标域结果。
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
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_ROOTS = {
    "DAN": PROJECT_ROOT / "outputs" / "DAN",
    "DSAN": PROJECT_ROOT / "outputs" / "DSAN",
    "DANN": PROJECT_ROOT / "outputs" / "DANN",
}

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_FT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "FT"
    / "finetune_full_stft_r0.10_summary.json"
)


def result_path(method: str, root: Path, source: str, target: str, seed: int) -> Path:
    pair_dir = root / f"{source}_to_{target}"
    if method == "DAN":
        run_dir = f"stft_dan_ssda_seed{seed}_r0.100"
    elif method == "DSAN":
        run_dir = f"stft_dsan_ssda_seed{seed}_r0.100"
    elif method == "DANN":
        run_dir = f"stft_ssda_seed{seed}_r0.10"
    else:
        raise ValueError(f"不支持的方法：{method}")
    return pair_dir / run_dir / "results.json"


def target_metrics(method: str, result: dict[str, Any]) -> tuple[float, float]:
    section = result["final"]["tgt_test"] if method == "DANN" else result["target_test"]
    accuracy = float(section["accuracy"])
    macro_f1 = float(section["macro_f1"])
    if not (math.isfinite(accuracy) and math.isfinite(macro_f1)):
        raise ValueError("accuracy 或 macro_f1 不是有限数值")
    return accuracy, macro_f1


def load_one(
    method: str, path: Path, source: str, target: str, expected_seed: int
) -> dict[str, float | int | str]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    actual = (str(result.get("source")), str(result.get("target")), int(result.get("seed", -1)))
    expected = (source, target, expected_seed)
    if actual != expected:
        raise ValueError(f"元数据不匹配：期望 {expected}，实际 {actual}")

    # 防止误把其他目标标签比例的结果混入 10% SSDA 汇总。
    if method == "DANN":
        ratio = float(result.get("hyper_params", {}).get("tgt_label_ratio", -1))
    else:
        ratio = float(result.get("config", {}).get("ratio", -1))
    if not math.isclose(ratio, 0.10, abs_tol=1e-9):
        raise ValueError(f"目标标签比例不是 0.10，而是 {ratio}")

    accuracy, macro_f1 = target_metrics(method, result)
    return {
        "seed": expected_seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "result_json": str(path),
    }


def summary_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def load_ft_rows(summary_path: Path) -> list[dict[str, Any]]:
    """读取 FT 汇总；每个 source->target 只选择 target 测试域。"""
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("seeds") != list(SEEDS):
        raise ValueError(f"FT seeds 应为 {list(SEEDS)}，实际为 {data.get('seeds')}")
    if not math.isclose(float(data.get("tgt_ratio", -1)), 0.10, abs_tol=1e-9):
        raise ValueError(f"FT 目标标签比例不是 0.10，而是 {data.get('tgt_ratio')}")

    rows: list[dict[str, Any]] = []
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            transfer = f"{source}->{target}"
            try:
                # 注意：方向下有四个测试域，这里只取与 target 同名的一项。
                metrics = data["summary"][transfer][target]
                accuracy = float(metrics["acc_mean"])
                macro_f1 = float(metrics["macroF1_mean"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"FT 汇总缺少有效的 {transfer}/{target} 指标") from exc
            if not (math.isfinite(accuracy) and math.isfinite(macro_f1)):
                raise ValueError(f"FT {transfer} 的目标域指标不是有限数值")
            rows.append(
                {
                    "method": "FT",
                    "transfer": transfer,
                    "source": source,
                    "target": target,
                    "n": len(SEEDS),
                    "missing_seeds": [],
                    "accuracy_mean": accuracy,
                    "accuracy_std": None,
                    "macro_f1_mean": macro_f1,
                    "macro_f1_std": None,
                    "runs": [],
                    "summary_json": str(summary_path),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 DAN/DSAN/DANN/FT 的 12 个迁移方向目标域指标"
    )
    parser.add_argument("--dan-root", type=Path, default=DEFAULT_ROOTS["DAN"])
    parser.add_argument("--dsan-root", type=Path, default=DEFAULT_ROOTS["DSAN"])
    parser.add_argument("--dann-root", type=Path, default=DEFAULT_ROOTS["DANN"])
    parser.add_argument("--ft-summary", type=Path, default=DEFAULT_FT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许用当前已有 seed 生成阶段性汇总；输出中的 n 和 missing_seeds 会标明完整性",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {
        "DAN": args.dan_root.resolve(),
        "DSAN": args.dsan_root.resolve(),
        "DANN": args.dann_root.resolve(),
    }

    rows: list[dict[str, Any]] = []
    missing_paths: list[Path] = []

    for method, root in roots.items():
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue

                runs: list[dict[str, float | int | str]] = []
                missing_seeds: list[int] = []
                for seed in SEEDS:
                    path = result_path(method, root, source, target, seed)
                    if not path.is_file():
                        missing_paths.append(path)
                        missing_seeds.append(seed)
                        continue
                    try:
                        runs.append(load_one(method, path, source, target, seed))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise RuntimeError(f"无法读取有效结果 {path}: {exc}") from exc

                acc_values = [float(run["accuracy"]) for run in runs]
                f1_values = [float(run["macro_f1"]) for run in runs]
                acc_mean, acc_std = summary_stats(acc_values)
                f1_mean, f1_std = summary_stats(f1_values)
                rows.append(
                    {
                        "method": method,
                        "transfer": f"{source}->{target}",
                        "source": source,
                        "target": target,
                        "n": len(runs),
                        "missing_seeds": missing_seeds,
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "macro_f1_mean": f1_mean,
                        "macro_f1_std": f1_std,
                        "runs": runs,
                    }
                )

    ft_summary = args.ft_summary.resolve()
    if not ft_summary.is_file():
        raise FileNotFoundError(f"FT 汇总文件不存在：{ft_summary}")
    try:
        rows.extend(load_ft_rows(ft_summary))
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取有效 FT 汇总 {ft_summary}: {exc}") from exc

    if missing_paths and not args.allow_incomplete:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:20])
        remainder = len(missing_paths) - min(20, len(missing_paths))
        extra = f"\n  ... 另外还有 {remainder} 个缺失文件" if remainder else ""
        raise SystemExit(
            f"结果尚未完整：缺少 {len(missing_paths)} 个 results.json。\n"
            f"{preview}{extra}\n"
            "训练完成后重新运行，或添加 --allow-incomplete 查看阶段性均值。"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "da_10pct_target_metrics_5seed.csv"
    json_path = args.out_dir / "da_10pct_target_metrics_5seed.json"

    csv_fields = [
        "method", "transfer", "source", "target", "n", "missing_seeds",
        "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row[key] for key in csv_fields}
            csv_row["missing_seeds"] = ",".join(map(str, row["missing_seeds"]))
            for key in ("accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"):
                value = csv_row[key]
                csv_row[key] = "" if value is None else f"{value:.6f}"
            writer.writerow(csv_row)

    payload = {
        "setting": "SSDA",
        "target_label_ratio": 0.10,
        "expected_seeds": list(SEEDS),
        "complete": not missing_paths,
        "missing_result_count": len(missing_paths),
        "method_roots": {method: str(root) for method, root in roots.items()},
        "ft_summary": str(ft_summary),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"完整性：{'完整（每个方向 5 个 seed）' if not missing_paths else f'阶段性结果，缺少 {len(missing_paths)} 个文件'}")


if __name__ == "__main__":
    main()
