#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run and summarize a DANN UDA/SSDA STFT labeled-target-ratio sweep.

One invocation runs 0%, 1%, 5%, 10%, and 20% labeled-target experiments over all
12 directed domain pairs and five random seeds.  Training is delegated to the
existing ``DANN_ssda`` implementation so its checkpoints and resume behavior
remain unchanged.  After each ratio, this script writes one plotting-ready
summary JSON aggregated from the 60 per-run ``results.json`` files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_SCRIPT = PROJECT_ROOT/ "DANN" / "DANN_ssda.py" 
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "DANN"
DEFAULT_SUMMARY_DIR =(PROJECT_ROOT / "outputs" / "DANN_target_label_ratio")

DOMAINS = ["base", "enam", "msh", "hlp"]
SEEDS = [0, 1, 2, 3, 4]
LABEL_RATIOS = [0.00, 0.01, 0.05, 0.10, 0.20]
METRIC_SECTIONS = [
    "src_test",
    "tgt_test",
    "src_test_at_tcal",
    "tgt_test_at_tcal",
    "tgt_test_at_trate",
]
THRESHOLD_KEYS = [
    "best_thresh",
    "best_thresh_ckpt",
    "best_thresh_calib",
    "rate_matched_thresh",
    "calibration_T",
]


def ratio_percent(ratio: float) -> int:
    percent = ratio * 100
    rounded = round(percent)
    if not math.isclose(percent, rounded, abs_tol=1e-9):
        raise ValueError(f"Ratio must map to an integer percentage: {ratio}")
    return int(rounded)


def pair_names() -> list[str]:
    return [f"{source}->{target}" for source in DOMAINS for target in DOMAINS if source != target]


def result_path(out_root: Path, source: str, target: str, seed: int, ratio: float) -> Path:
    # DANN_ssda names the zero-label UDA run without an r0.00 suffix.
    run_name = (
        f"stft_dann_seed{seed}"
        if math.isclose(ratio, 0.0, abs_tol=1e-12)
        else f"stft_ssda_seed{seed}_r{ratio:.2f}"
    )
    return out_root / f"{source}_to_{target}" / run_name / "results.json"


def load_result(path: Path, expected_ratio: float) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    required = {"source", "target", "seed", "modality", "final", "hyper_params"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Missing keys {sorted(missing)} in {path}")
    if str(result["modality"]).upper() != "STFT":
        raise ValueError(f"Expected STFT result, got {result['modality']!r}: {path}")
    actual_ratio = float(result["hyper_params"].get("tgt_label_ratio", -1))
    if not math.isclose(actual_ratio, expected_ratio, abs_tol=1e-9):
        raise ValueError(
            f"Ratio mismatch in {path}: expected {expected_ratio:.2f}, got {actual_ratio}"
        )
    return result


def stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        raise ValueError("Cannot aggregate an empty metric list")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95": round(ci95, 6),
        "n": n,
    }


def numeric_metrics(runs: list[dict[str, Any]], section: str) -> dict[str, dict[str, float | int]]:
    metric_names: list[str] = []
    for run in runs:
        section_data = run["final"].get(section)
        if not isinstance(section_data, dict):
            continue
        for key, value in section_data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in metric_names:
                metric_names.append(key)

    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        values = []
        for run in runs:
            value = run["final"].get(section, {}).get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            aggregate[metric] = stats(values)
    return aggregate


def compact_run(result: dict[str, Any], result_json: Path) -> dict[str, Any]:
    final = result["final"]
    metrics = {
        section: final[section]
        for section in METRIC_SECTIONS
        if isinstance(final.get(section), dict)
    }
    thresholds = {
        key: final[key]
        for key in THRESHOLD_KEYS
        if isinstance(final.get(key), (int, float)) and not isinstance(final.get(key), bool)
    }
    return {
        "seed": int(result["seed"]),
        "source": result["source"],
        "target": result["target"],
        "mode": result.get("mode", "DANN_SSDA"),
        "modality": result["modality"],
        "target_label_ratio": float(result["hyper_params"]["tgt_label_ratio"]),
        "train_time_sec": result.get("train_time_sec"),
        "sizes": result.get("sizes", {}),
        "result_json": str(result_json),
        "metrics": metrics,
        "thresholds": thresholds,
    }


def aggregate_ratio(out_root: Path, summary_dir: Path, ratio: float) -> Path:
    percent = ratio_percent(ratio)
    summary: dict[str, Any] = {}
    all_paths: list[Path] = []

    for pair in pair_names():
        source, target = pair.split("->")
        loaded: list[tuple[dict[str, Any], Path]] = []
        for seed in SEEDS:
            path = result_path(out_root, source, target, seed, ratio)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing completed run for {percent}%: {path}\n"
                    "Re-run without --summarize-only to train/resume missing experiments."
                )
            loaded.append((load_result(path, ratio), path))
            all_paths.append(path)

        loaded.sort(key=lambda item: int(item[0]["seed"]))
        results = [item[0] for item in loaded]
        aggregate = {
            section: numeric_metrics(results, section)
            for section in METRIC_SECTIONS
            if any(isinstance(result["final"].get(section), dict) for result in results)
        }
        aggregate["thresholds"] = {
            key: stats([float(result["final"][key]) for result in results])
            for key in THRESHOLD_KEYS
            if all(isinstance(result["final"].get(key), (int, float)) for result in results)
        }
        summary[pair] = {
            "num_runs": len(results),
            "seeds": [int(result["seed"]) for result in results],
            "aggregate": aggregate,
            "runs": [compact_run(result, path) for result, path in loaded],
        }

    payload = {
        "method": "DANN",
        "setting": "UDA" if percent == 0 else "SSDA",
        "target_label_ratio": ratio,
        "target_label_percent": percent,
        "modality": "STFT",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(out_root),
        "num_result_files": len(all_paths),
        "pairs_run": pair_names(),
        "seeds_expected": SEEDS,
        "summary": summary,
    }

    summary_dir.mkdir(parents=True, exist_ok=True)
    mode_name = "uda" if percent == 0 else "ssda"
    output = summary_dir / f"dann_stft_{mode_name}_{percent}pct_summary.json"
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(output)
    return output


def train_ratio(args: argparse.Namespace, ratio: float) -> None:
    # Restrict visibility before the trainer imports torch.  Inside that
    # one-visible-GPU process, CUDA device 0 is the requested physical GPU.
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        args.python,
        str(args.train_script),
        "--out-root", str(args.out_root),
        "--cross-train-all", "1",
        "--semi", "0" if math.isclose(ratio, 0.0, abs_tol=1e-12) else "1",
        "--tgt-label-ratio", f"{ratio:.2f}",
        "--seeds", *[str(seed) for seed in SEEDS],
        "--domains", *DOMAINS,
        "--epochs", str(args.epochs),
        "--num-workers", str(args.num_workers),
    ]
    if args.trainer_extra_args:
        command.extend(args.trainer_extra_args)
    mode_name = "UDA" if math.isclose(ratio, 0.0, abs_tol=1e-12) else "SSDA"
    print(f"\n[SWEEP] Training/resuming DANN-{mode_name} at {ratio_percent(ratio)}% labeled target data", flush=True)
    print("[COMMAND] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DANN STFT at 0/1/5/10/20% and create one JSON per ratio."
    )
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not train; rebuild the five JSON summaries from completed results.",
    )
    parser.add_argument(
        "--trainer-extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to DANN_ssda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.train_script = args.train_script.resolve()
    args.out_root = args.out_root.resolve()
    args.summary_dir = args.summary_dir.resolve()
    if not args.train_script.is_file():
        raise FileNotFoundError(f"DANN training script not found: {args.train_script}")
    if args.epochs < 1 or args.num_workers < 0:
        raise ValueError("epochs must be >= 1 and num-workers must be >= 0")

    outputs = []
    for ratio in LABEL_RATIOS:
        if not args.summarize_only:
            train_ratio(args, ratio)
        output = aggregate_ratio(args.out_root, args.summary_dir, ratio)
        outputs.append(output)
        print(f"[SUMMARY] {ratio_percent(ratio)}% -> {output}", flush=True)

    print("\n[DONE] All ratio summaries are ready:", flush=True)
    for output in outputs:
        print(f"  - {output}", flush=True)


if __name__ == "__main__":
    main()
