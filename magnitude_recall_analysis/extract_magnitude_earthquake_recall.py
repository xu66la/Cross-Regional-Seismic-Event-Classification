#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract magnitude-stratified earthquake recall for 12 transfer tasks.

Default scope:

* methods: Full Fine-tuning, DANN, DAN, DSAN;
* tasks: all 12 directed transfers among BASE/MSH/ENAM/HLP;
* seeds: 0, 1, 2, 3, 4;
* labeled target ratio: 10%;
* magnitude bins: [0,1), [1,2), [2,3), [3,+inf).

The script performs a complete checkpoint preflight, extracts per-sample target
test predictions to reusable NPZ files, verifies IDs and labels against the
current target dataset, joins each sample to manifest_final.csv and events.json,
computes seed-wise earthquake recall, averages it over seeds, and writes the
``gain_connect.csv`` consumed by the two plotting scripts in this directory.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "magnitude_recall_analysis"
)

MODEL_HELPER_PATH = (
    PROJECT_ROOT
    / "negative_transfer_analysis"
    / "extract_plot_base_msh_to_enam_multiseed.py"
)

DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DOMAINS = ("base", "msh", "enam", "hlp")
METHODS = ("full_ft", "dan", "dann", "dsan")
METHOD_LABELS = {
    "full_ft": "Full Fine-tuning",
    "dan": "DAN",
    "dann": "DANN",
    "dsan": "DSAN",
}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
MAGNITUDE_BINS = ("0-1", "1-2", "2-3", ">=3")
TASKS = tuple(
    f"{source}_to_{target}"
    for source in DOMAINS
    for target in DOMAINS
    if source != target
)


def load_model_helper():
    if not MODEL_HELPER_PATH.is_file():
        raise FileNotFoundError(f"Model helper not found: {MODEL_HELPER_PATH}")
    spec = importlib.util.spec_from_file_location("manuscript_confusion_helper", MODEL_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model helper: {MODEL_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_HELPER = load_model_helper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 5-seed, magnitude-stratified earthquake recall for domain transfer."
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0"
    )
    parser.add_argument(
        "--overwrite-npz",
        action="store_true",
        help="Recompute predictions even if a matching cached NPZ exists.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check all required checkpoint/data/metadata files without inference.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def domain_root(domain: str) -> Path:
    upper = domain.upper()
    return MANUSCRIPT_ROOT / upper / f"processing_{upper}_outputs"


def target_test_path(domain: str) -> Path:
    root = domain_root(domain) / "legacy_dataset"
    expected = list(root.glob(f"{domain}_dataset_*_test_STFT_raw.pt"))
    if len(expected) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {domain} test STFT file in {root}, found {len(expected)}"
        )
    return expected[0]


def metadata_paths(domain: str) -> tuple[Path, Path]:
    root = domain_root(domain)
    return root / "manifest_final.csv", root / "events.json"


def split_task(task: str) -> tuple[str, str]:
    source, target = task.split("_to_")
    return source, target


def preflight(
    tasks: list[str], seeds: list[int]
) -> dict[tuple[str, str, int], Path]:
    targets = sorted({split_task(task)[1] for task in tasks})
    problems: list[str] = []
    for target in targets:
        try:
            test_path = target_test_path(target)
        except FileNotFoundError as exc:
            problems.append(str(exc))
            continue
        manifest_path, events_path = metadata_paths(target)
        for required in (test_path, manifest_path, events_path):
            if not required.is_file():
                problems.append(f"Missing target data/metadata: {required}")

    checkpoints: dict[tuple[str, str, int], Path] = {}
    for task in tasks:
        for method in METHODS:
            for seed in seeds:
                try:
                    path = MODEL_HELPER.checkpoint_path(task, method, seed)
                except Exception as exc:
                    problems.append(f"{task} | {method} | seed={seed}: {exc}")
                    continue
                if path.is_file():
                    checkpoints[(task, method, seed)] = path
                else:
                    tried = MODEL_HELPER.checkpoint_candidates(task, method, seed)
                    problems.append(
                        f"Missing {task} | {method} | seed={seed}; tried: "
                        + "; ".join(str(candidate) for candidate in tried)
                    )
    if problems:
        preview = "\n".join(f"  - {problem}" for problem in problems[:30])
        remainder = len(problems) - min(len(problems), 30)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(f"Preflight failed with {len(problems)} problem(s):\n{preview}{suffix}")
    return checkpoints


def canonical_waveform_name(sample_id: str) -> str:
    name = Path(str(sample_id)).name
    suffix = "_STFT_raw.npy"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected STFT sample ID: {sample_id}")
    return name[: -len(suffix)] + ".npy"


def magnitude_bin(value: float) -> str:
    if 0.0 <= value < 1.0:
        return "0-1"
    if 1.0 <= value < 2.0:
        return "1-2"
    if 2.0 <= value < 3.0:
        return "2-3"
    if value >= 3.0:
        return ">=3"
    raise ValueError(f"Magnitude must be finite and non-negative, got {value}")


def load_target_bundle(
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Path]:
    test_path = target_test_path(target)
    obj = torch.load(test_path, map_location="cpu")
    if not isinstance(obj, dict) or not {"data", "labels_num", "ids"}.issubset(obj):
        raise ValueError(f"Expected data/labels_num/ids dictionary: {test_path}")
    raw_x = MODEL_HELPER.tensor_from_data(obj["data"])
    labels = torch.as_tensor(obj["labels_num"]).long().view(-1)
    sample_ids = np.asarray([str(item) for item in obj["ids"]], dtype=str)
    if len(raw_x) != len(labels) or len(labels) != len(sample_ids):
        raise ValueError(
            f"Target length mismatch for {target}: x={len(raw_x)}, y={len(labels)}, ids={len(sample_ids)}"
        )
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"Duplicate target sample IDs in {test_path}")

    manifest_path, events_path = metadata_paths(target)
    manifest = pd.read_csv(manifest_path)
    required_manifest = {"event_id", "event_type", "split", "waveform"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"Missing manifest columns {sorted(missing_manifest)}: {manifest_path}")
    manifest = manifest[manifest["split"].astype(str).str.lower() == "test"].copy()
    manifest["canonical_filename"] = manifest["waveform"].map(
        lambda value: Path(str(value)).name
    )
    if manifest["canonical_filename"].duplicated().any():
        raise ValueError(f"Duplicate test waveform filenames in {manifest_path}")
    manifest = manifest.set_index("canonical_filename", drop=False)

    canonical_names = np.asarray(
        [canonical_waveform_name(sample_id) for sample_id in sample_ids], dtype=str
    )
    missing_names = sorted(set(canonical_names) - set(manifest.index))
    extra_names = sorted(set(manifest.index) - set(canonical_names))
    if missing_names or extra_names:
        raise ValueError(
            f"Target IDs do not match test manifest for {target}; "
            f"missing={missing_names[:5]}, extra={extra_names[:5]}"
        )
    aligned_manifest = manifest.loc[canonical_names].reset_index(drop=True)

    label_map = {"earthquake": 0, "explosion": 1}
    metadata_labels = aligned_manifest["event_type"].astype(str).str.lower().map(label_map)
    if metadata_labels.isna().any():
        raise ValueError(f"Unknown event_type in {manifest_path}")
    if not np.array_equal(metadata_labels.to_numpy(dtype=int), labels.numpy()):
        raise ValueError(f"Target labels do not match manifest event_type for {target}")

    with events_path.open("r", encoding="utf-8") as handle:
        events = json.load(handle)
    magnitudes: list[float] = []
    event_ids: list[str] = []
    for event_id in aligned_manifest["event_id"].astype(str):
        if event_id not in events:
            raise KeyError(f"Event {event_id} from manifest not found in {events_path}")
        value = events[event_id].get("magnitude")
        try:
            magnitude = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid magnitude for event {event_id}: {value}") from exc
        if not math.isfinite(magnitude) or magnitude < 0:
            raise ValueError(f"Invalid magnitude for event {event_id}: {magnitude}")
        event_ids.append(event_id)
        magnitudes.append(magnitude)
    magnitude_values = np.asarray(magnitudes, dtype=np.float64)
    magnitude_bins = np.asarray([magnitude_bin(value) for value in magnitude_values], dtype=str)

    sample_map = pd.DataFrame(
        {
            "target_domain": target,
            "sample_id": sample_ids,
            "canonical_filename": canonical_names,
            "event_id": event_ids,
            "event_type": aligned_manifest["event_type"].astype(str).to_numpy(),
            "true_label": labels.numpy().astype(int),
            "magnitude": magnitude_values,
            "magnitude_bin": magnitude_bins,
        }
    )
    return raw_x, labels, sample_ids, magnitude_values, magnitude_bins, sample_map, test_path


def validate_checkpoint_identity(
    checkpoint: Mapping[str, Any], task: str, method: str, seed: int
) -> None:
    expected_source, expected_target = split_task(task)
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), Mapping) else {}
    observed = {
        "source": checkpoint.get("source", config.get("source")),
        "target": checkpoint.get("target", config.get("target")),
        "seed": checkpoint.get("seed", config.get("seed")),
    }
    expected = {"source": expected_source, "target": expected_target, "seed": seed}
    for key, expected_value in expected.items():
        value = observed[key]
        if value is not None and str(value).lower() != str(expected_value).lower():
            raise ValueError(
                f"Checkpoint identity mismatch for {task}/{method}/seed{seed}: "
                f"{key}={value}, expected {expected_value}"
            )
    ratio = checkpoint.get("tgt_ratio", checkpoint.get("tgt_label_ratio", config.get("ratio")))
    if ratio is not None and not math.isclose(float(ratio), 0.10, abs_tol=1e-9):
        raise ValueError(
            f"Checkpoint is not a 10% target-label run for {task}/{method}/seed{seed}: ratio={ratio}"
        )


def npz_scalar(z: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def cache_matches(npz_path: Path, checkpoint_path: Path, test_path: Path) -> bool:
    if not npz_path.is_file():
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            required = {
                "target_labels",
                "target_predictions",
                "target_probabilities",
                "target_sample_ids",
                "target_magnitudes",
                "target_magnitude_bins",
                "checkpoint_path",
                "checkpoint_size",
                "checkpoint_mtime_ns",
                "target_test_path",
                "target_test_size",
                "target_test_mtime_ns",
            }
            if not required.issubset(z.files):
                return False
            checkpoint_stat = checkpoint_path.stat()
            test_stat = test_path.stat()
            return (
                str(npz_scalar(z, "checkpoint_path")) == str(checkpoint_path)
                and int(npz_scalar(z, "checkpoint_size")) == checkpoint_stat.st_size
                and int(npz_scalar(z, "checkpoint_mtime_ns")) == checkpoint_stat.st_mtime_ns
                and str(npz_scalar(z, "target_test_path")) == str(test_path)
                and int(npz_scalar(z, "target_test_size")) == test_stat.st_size
                and int(npz_scalar(z, "target_test_mtime_ns")) == test_stat.st_mtime_ns
            )
    except Exception:
        return False


def load_cached(
    npz_path: Path,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    magnitudes: np.ndarray,
    magnitude_bins: np.ndarray,
) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as z:
        result = {
            "labels": z["target_labels"].astype(np.int64),
            "predictions": z["target_predictions"].astype(np.int64),
            "probabilities": z["target_probabilities"].astype(np.float64),
            "sample_ids": z["target_sample_ids"].astype(str),
            "magnitudes": z["target_magnitudes"].astype(np.float64),
            "magnitude_bins": z["target_magnitude_bins"].astype(str),
            "threshold": float(npz_scalar(z, "threshold")),
        }
    checks = {
        "labels": np.array_equal(result["labels"], labels),
        "sample IDs/order": np.array_equal(result["sample_ids"], sample_ids),
        "magnitudes": np.array_equal(result["magnitudes"], magnitudes),
        "magnitude bins": np.array_equal(result["magnitude_bins"], magnitude_bins),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Cached NPZ does not match current target data ({', '.join(failed)}): {npz_path}")
    return result


@torch.inference_mode()
def extract_prediction(
    task: str,
    method: str,
    seed: int,
    checkpoint_path: Path,
    raw_x: torch.Tensor,
    labels_tensor: torch.Tensor,
    sample_ids: np.ndarray,
    magnitudes: np.ndarray,
    magnitude_bins: np.ndarray,
    test_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> tuple[dict[str, Any], Path, bool]:
    npz_path = output_dir / "npz" / f"{task}_{method}_seed{seed}_target_predictions_magnitude.npz"
    labels = labels_tensor.numpy().astype(np.int64)
    if not overwrite and cache_matches(npz_path, checkpoint_path, test_path):
        return (
            load_cached(npz_path, labels, sample_ids, magnitudes, magnitude_bins),
            npz_path,
            True,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected dictionary checkpoint: {checkpoint_path}")
    validate_checkpoint_identity(checkpoint, task, method, seed)
    model, mean, std, threshold, strict_greater = MODEL_HELPER.checkpoint_contract(
        checkpoint, method, task, seed
    )
    model.to(device).eval()
    dataset = MODEL_HELPER.TargetDataset(
        raw_x,
        labels_tensor,
        sample_ids,
        mean,
        std,
        apply_dann_log1p=(method == "dann"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    labels_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    extracted_ids: list[str] = []
    for inputs, batch_labels, batch_ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and method == "full_ft"),
        ):
            output = model(inputs)
        logits = output[0] if isinstance(output, tuple) else output
        probability_parts.append(torch.sigmoid(logits).view(-1).float().cpu().numpy())
        labels_parts.append(batch_labels.view(-1).cpu().numpy().astype(np.int64))
        extracted_ids.extend(str(item) for item in batch_ids)

    extracted_labels = np.concatenate(labels_parts)
    probabilities = np.concatenate(probability_parts).astype(np.float64)
    predictions = (
        probabilities > threshold if strict_greater else probabilities >= threshold
    ).astype(np.int64)
    extracted_ids_array = np.asarray(extracted_ids, dtype=str)
    if not np.array_equal(extracted_labels, labels):
        raise RuntimeError(f"Inference changed target labels: {task}/{method}/seed{seed}")
    if not np.array_equal(extracted_ids_array, sample_ids):
        raise RuntimeError(f"Inference changed target sample order: {task}/{method}/seed{seed}")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_stat = checkpoint_path.stat()
    test_stat = test_path.stat()
    np.savez_compressed(
        npz_path,
        target_labels=extracted_labels,
        target_predictions=predictions,
        target_probabilities=probabilities,
        target_sample_ids=extracted_ids_array,
        target_magnitudes=magnitudes,
        target_magnitude_bins=magnitude_bins,
        task=task,
        method=method,
        seed=seed,
        threshold=threshold,
        comparison=(">" if strict_greater else ">="),
        normalization_mean=mean,
        normalization_std=std,
        checkpoint_path=str(checkpoint_path),
        checkpoint_size=checkpoint_stat.st_size,
        checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns,
        target_test_path=str(test_path),
        target_test_size=test_stat.st_size,
        target_test_mtime_ns=test_stat.st_mtime_ns,
    )
    result = {
        "labels": extracted_labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "sample_ids": extracted_ids_array,
        "magnitudes": magnitudes,
        "magnitude_bins": magnitude_bins,
        "threshold": threshold,
    }
    model.to("cpu")
    del model, checkpoint, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result, npz_path, False


def seedwise_recall_rows(
    task: str, method: str, seed: int, result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    labels = np.asarray(result["labels"], dtype=np.int64)
    predictions = np.asarray(result["predictions"], dtype=np.int64)
    bins = np.asarray(result["magnitude_bins"], dtype=str)
    rows: list[dict[str, Any]] = []
    for bin_label in MAGNITUDE_BINS:
        mask = (labels == 0) & (bins == bin_label)
        sample_count = int(mask.sum())
        correct_count = int((predictions[mask] == 0).sum()) if sample_count else 0
        rows.append(
            {
                "task": task,
                "source": split_task(task)[0],
                "target": split_task(task)[1],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "magnitude_bin": bin_label,
                "earthquake_samples": sample_count,
                "earthquake_correct": correct_count,
                "earthquake_recall": correct_count / sample_count if sample_count else np.nan,
            }
        )
    return rows


def build_gain_connect(summary: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [tasks, MAGNITUDE_BINS], names=["task", "magnitude_bin"]
    )
    wide = summary.pivot(
        index=["task", "magnitude_bin"],
        columns="method",
        values="earthquake_recall_mean",
    ).reindex(index)
    for method in METHODS:
        if method not in wide.columns:
            wide[method] = np.nan
    wide = wide[list(METHODS)].reset_index()
    for method in ("dan", "dann", "dsan"):
        difference = f"{method}_minus_full_ft"
        relative = f"{method}_relative_gain_percent"
        wide[difference] = wide[method] - wide["full_ft"]
        wide[relative] = np.where(
            wide["full_ft"].notna() & (wide["full_ft"] != 0),
            100.0 * wide[difference] / wide["full_ft"],
            np.nan,
        )
    return wide


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be >= 1 and num-workers must be >= 0")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and contain no duplicates")

    checkpoints = preflight(args.tasks, args.seeds)
    print(f"[PREFLIGHT] {len(checkpoints)} checkpoints complete", flush=True)
    if args.preflight_only:
        print("[DONE] Preflight passed.", flush=True)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[DEVICE] {device}", flush=True)

    seedwise_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    sample_maps: list[pd.DataFrame] = []

    tasks_by_target = {
        target: [task for task in args.tasks if split_task(task)[1] == target]
        for target in DOMAINS
    }
    for target, target_tasks in tasks_by_target.items():
        if not target_tasks:
            continue
        print(f"\n[LOAD TARGET] {target}", flush=True)
        (
            raw_x,
            labels_tensor,
            sample_ids,
            magnitudes,
            magnitude_bins,
            sample_map,
            test_path,
        ) = load_target_bundle(target)
        sample_maps.append(sample_map)
        print(f"[TARGET] {target}: n={len(labels_tensor)}, data={test_path}", flush=True)

        for task in target_tasks:
            for method in METHODS:
                reference_ids: np.ndarray | None = None
                reference_labels: np.ndarray | None = None
                for seed in args.seeds:
                    checkpoint_path = checkpoints[(task, method, seed)]
                    print(
                        f"[INFER] {task} | {method} | seed={seed} | {checkpoint_path}",
                        flush=True,
                    )
                    result, npz_path, cached = extract_prediction(
                        task,
                        method,
                        seed,
                        checkpoint_path,
                        raw_x,
                        labels_tensor,
                        sample_ids,
                        magnitudes,
                        magnitude_bins,
                        test_path,
                        output_dir,
                        device,
                        args.batch_size,
                        args.num_workers,
                        args.overwrite_npz,
                    )
                    if reference_ids is None:
                        reference_ids = np.asarray(result["sample_ids"])
                        reference_labels = np.asarray(result["labels"])
                    elif not np.array_equal(result["sample_ids"], reference_ids):
                        raise ValueError(f"Sample order differs across seeds: {task}/{method}/seed{seed}")
                    elif not np.array_equal(result["labels"], reference_labels):
                        raise ValueError(f"Labels differ across seeds: {task}/{method}/seed{seed}")
                    seedwise_rows.extend(seedwise_recall_rows(task, method, seed, result))
                    checkpoint_rows.append(
                        {
                            "task": task,
                            "method": method,
                            "seed": seed,
                            "checkpoint_path": str(checkpoint_path),
                            "npz_path": str(npz_path),
                            "threshold": result["threshold"],
                            "used_cached_npz": cached,
                        }
                    )
        del raw_x, labels_tensor, sample_ids, magnitudes, magnitude_bins, sample_map
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seedwise = pd.DataFrame(seedwise_rows)
    summary = (
        seedwise.groupby(
            ["task", "source", "target", "method", "method_label", "magnitude_bin"],
            as_index=False,
            dropna=False,
        )
        .agg(
            earthquake_samples_per_seed=("earthquake_samples", "first"),
            seeds=("seed", "nunique"),
            earthquake_recall_mean=("earthquake_recall", "mean"),
            earthquake_recall_std=("earthquake_recall", "std"),
        )
    )
    gain_connect = build_gain_connect(summary, args.tasks)

    seedwise_path = output_dir / "earthquake_recall_by_magnitude_seedwise.csv"
    summary_path = output_dir / "earthquake_recall_by_magnitude_mean_std.csv"
    gain_path = output_dir / "gain_connect.csv"
    sample_map_path = output_dir / "target_test_sample_magnitude_map.csv"
    checkpoints_path = output_dir / "used_checkpoints_magnitude_recall.csv"
    seedwise.to_csv(seedwise_path, index=False)
    summary.to_csv(summary_path, index=False)
    gain_connect.to_csv(gain_path, index=False)
    pd.concat(sample_maps, ignore_index=True).drop_duplicates(
        ["target_domain", "sample_id"]
    ).to_csv(sample_map_path, index=False)
    pd.DataFrame(checkpoint_rows).to_csv(checkpoints_path, index=False)

    print("\n[DONE] Magnitude-recall data are ready:", flush=True)
    for path in (seedwise_path, summary_path, gain_path, sample_map_path, checkpoints_path):
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main()
