#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract predictions and draw multi-seed confusion matrices.

The default analysis covers BASE->ENAM and MSH->ENAM at 10% labeled target
data for Full FT, DANN, DAN, and DSAN.  For every task/method it:

1. loads the five manuscript checkpoints (seeds 0..4);
2. runs inference on the ENAM target-test set with the method's original
   preprocessing and decision threshold;
3. stores per-seed labels, predictions, probabilities, and sample IDs in NPZ;
4. strictly verifies target sample order and labels across all seeds/methods;
5. concatenates the five prediction sets, computes a confusion matrix and
   Macro-F1, and writes a 2x2 four-method figure.

The script performs a complete checkpoint preflight before loading test data,
so a missing model cannot silently produce an incomplete formal figure.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = Path("/path/to/US_EQ_EX")

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "negative_transfer_analysis"
)

ENAM_TEST_PT = (
    DATA_ROOT
    / "ENAM"
    / "processing_ENAM_outputs"
    / "legacy_dataset"
    / "enam_dataset_zrt_waveforms_ZRT_DanFenLiang_40HZ_90S_EventFiltered_test_STFT_raw.pt"
)


TASKS = {
    "base_to_enam": {"source": "base", "target": "enam", "label": "BASE→ENAM"},
    "msh_to_enam": {"source": "msh", "target": "enam", "label": "MSH→ENAM"},
}
METHODS = ("full_ft", "dann", "dan", "dsan")
METHOD_LABELS = {
    "full_ft": "Full Fine-tuning",
    "dann": "DANN",
    "dan": "DAN",
    "dsan": "DSAN",
}
PANEL_LETTERS = {"full_ft": "a", "dann": "b", "dan": "c", "dsan": "d"}
CLASS_IDS = (0, 1)
CLASS_NAMES = ("earthquake", "explosion")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract five-seed predictions and draw BASE/MSH->ENAM confusion matrices."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASKS),
        default=list(TASKS),
        help="Transfer directions to process (default: both).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Seeds to concatenate."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a CUDA device such as cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--overwrite-npz",
        action="store_true",
        help="Recompute predictions even when a matching cached NPZ exists.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only check checkpoint/data availability; do not run inference.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device

def checkpoint_candidates(task: str, method: str, seed: int) -> list[Path]:

    if method == "full_ft":
        return [
            PROJECT_ROOT
            / "outputs"
            / "FT"
            / task
            / f"stft_ftfull_r0.10_seed{seed}"
            / f"{task}_best_ftfull.pth"
        ]

    if method == "dann":
        root = (
            PROJECT_ROOT
            / "outputs"
            / "DANN"
            / task
        )
        return [
            root
            / f"stft_ssda_seed{seed}_r{ratio}"
            / "best.pth"
            for ratio in ("0.10", "0.100", "0.1")
        ]

    if method == "dan":
        root = (
            PROJECT_ROOT
            / "outputs"
            / "DAN"
            / task
        )
        return [
            root
            / f"stft_dan_ssda_seed{seed}_r{ratio}"
            / "best.pth"
            for ratio in ("0.100", "0.10", "0.1")
        ]

    if method == "dsan":
        root = (
            PROJECT_ROOT
            / "outputs"
            / "DSAN"
            / task
        )
        return [
            root
            / f"stft_dsan_ssda_seed{seed}_r{ratio}"
            / "best.pth"
            for ratio in ("0.100", "0.10", "0.1")
        ]

    raise ValueError(f"Unknown method: {method}")


def checkpoint_path(task: str, method: str, seed: int) -> Path:
    candidates = checkpoint_candidates(task, method, seed)
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple checkpoints match {task}/{method}/seed{seed}:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    return existing[0] if existing else candidates[0]


def preflight(tasks: list[str], seeds: list[int]) -> dict[tuple[str, str, int], Path]:
    if not ENAM_TEST_PT.is_file():
        raise FileNotFoundError(f"ENAM target-test data not found: {ENAM_TEST_PT}")
    resolved: dict[tuple[str, str, int], Path] = {}
    missing: list[str] = []
    for task in tasks:
        for method in METHODS:
            for seed in seeds:
                path = checkpoint_path(task, method, seed)
                if path.is_file():
                    resolved[(task, method, seed)] = path
                else:
                    tried = checkpoint_candidates(task, method, seed)
                    missing.append(
                        f"{task} | {method} | seed={seed}\n"
                        + "\n".join(f"      tried: {candidate}" for candidate in tried)
                    )
    if missing:
        raise FileNotFoundError(
            "Cannot create complete four-method confusion matrices; missing checkpoints:\n  - "
            + "\n  - ".join(missing)
        )
    return resolved


def tensor_from_data(raw: Any) -> torch.Tensor:
    if isinstance(raw, list):
        x = torch.stack([torch.as_tensor(item) for item in raw])
    else:
        x = torch.as_tensor(raw)
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, 256, 256):
        raise ValueError(f"Expected test x=[N,1,256,256], got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("ENAM target-test data contains non-finite values")
    return x.contiguous()


def load_target_test() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    obj = torch.load(ENAM_TEST_PT, map_location="cpu")
    if not isinstance(obj, dict) or "data" not in obj or "labels_num" not in obj:
        raise ValueError(f"Expected a data/labels_num dictionary: {ENAM_TEST_PT}")
    x = tensor_from_data(obj["data"])
    labels = torch.as_tensor(obj["labels_num"]).long().view(-1)
    if len(x) != len(labels):
        raise ValueError(f"Data/label length mismatch: {len(x)} vs {len(labels)}")
    if not set(labels.unique().tolist()).issubset(CLASS_IDS):
        raise ValueError(f"Target labels must be 0/1, got {labels.unique().tolist()}")
    raw_ids = obj.get("ids")
    if raw_ids is None:
        raise ValueError(f"Target-test file has no 'ids' field: {ENAM_TEST_PT}")
    sample_ids = np.asarray([str(item) for item in raw_ids], dtype=str)
    if len(sample_ids) != len(labels):
        raise ValueError(f"ID/label length mismatch: {len(sample_ids)} vs {len(labels)}")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("ENAM target-test sample IDs are not unique")
    return x, labels, sample_ids


class TargetDataset(Dataset):
    def __init__(
        self,
        raw_x: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: np.ndarray,
        mean: float,
        std: float,
        apply_dann_log1p: bool,
    ) -> None:
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization: mean={mean}, std={std}")
        self.raw_x = raw_x
        self.labels = labels
        self.sample_ids = sample_ids
        self.mean = float(mean)
        self.std = float(std)
        self.apply_dann_log1p = apply_dann_log1p

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        x = self.raw_x[index]
        # This exactly follows the current manuscript DANN trainer.  The other
        # three trainers consume the pre-log1p artifact directly.
        if self.apply_dann_log1p:
            x = torch.log1p(torch.clamp(x, min=0.0))
        x = (x - self.mean) / (self.std + 1e-6)
        return x, self.labels[index], str(self.sample_ids[index])


class FTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv3, self.bn3 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.pool, self.dropout = nn.MaxPool2d(2, 2), nn.Dropout(0.5)
        self.flat_dim = 128 * 32 * 32
        self.reduce_fc, self.out_fc = nn.Linear(self.flat_dim, 128), nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x.flatten(1))
        return self.out_fc(torch.relu(self.reduce_fc(x)))


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DAFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(1, 32, 3, padding=1), group_norm(32)
        self.c2, self.n2 = nn.Conv2d(32, 64, 3, padding=1), group_norm(64)
        self.c3, self.n3 = nn.Conv2d(64, 128, 3, padding=1), group_norm(128)
        self.pool, self.drop = nn.MaxPool2d(2), nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.n1(self.c1(x))))
        x = self.pool(torch.relu(self.n2(self.c2(x))))
        x = self.pool(torch.relu(self.n3(self.c3(x))))
        return self.drop(x.flatten(1))


class DANLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = DAFeatures()
        self.reduce = nn.Linear(128 * 32 * 32, 128)
        self.cls = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.reduce(self.feature(x))
        return self.cls(features), features


class DANNFeatures(nn.Module):
    def __init__(self, use_group_norm: bool) -> None:
        super().__init__()
        norm = group_norm if use_group_norm else nn.BatchNorm2d
        self.conv1, self.bn1 = nn.Conv2d(1, 32, 3, padding=1), norm(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), norm(64)
        self.conv3, self.bn3 = nn.Conv2d(64, 128, 3, padding=1), norm(128)
        self.pool, self.dropout = nn.MaxPool2d(2, 2), nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        return self.dropout(x.flatten(1))


class DANNModel(nn.Module):
    def __init__(self, use_group_norm: bool) -> None:
        super().__init__()
        self.feature = DANNFeatures(use_group_norm)
        self.reduce = nn.Linear(128 * 32 * 32, 128)
        self.cls_head = nn.Linear(128, 1)
        self.domain_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.reduce(self.feature(x))
        return self.cls_head(features), self.domain_head(features)


def strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    return dict(state)


def checkpoint_contract(
    checkpoint: Mapping[str, Any], method: str, task: str, seed: int
) -> tuple[nn.Module, float, float, float, bool]:
    state_raw = checkpoint.get("model")
    if not isinstance(state_raw, Mapping):
        raise ValueError(f"Checkpoint has no model state dictionary: {task}/{method}/seed{seed}")
    state = strip_module_prefix(state_raw)

    if method == "full_ft":
        model = FTCNN()
        mean, std, threshold, strict_greater = checkpoint["mean"], checkpoint["std"], 0.5, True
    elif method == "dann":
        use_group_norm = not any(key.endswith("running_mean") for key in state)
        model = DANNModel(use_group_norm=use_group_norm)
        mean, std = checkpoint["src_mean"], checkpoint["src_std"]
        threshold, strict_greater = checkpoint["best_thresh"], False
    elif method in {"dan", "dsan"}:
        model = DANLike()
        mean, std = checkpoint["src_mean"], checkpoint["src_std"]
        threshold, strict_greater = checkpoint["best_thresh"], False
    else:
        raise ValueError(method)

    model.load_state_dict(state, strict=True)
    return model, float(mean), float(std), float(threshold), strict_greater


def scalar_from_npz(z: Mapping[str, Any], key: str) -> Any:
    value = z[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def cached_npz_matches(path: Path, checkpoint_path_: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"target_labels", "target_predictions", "target_probabilities", "target_sample_ids",
                        "checkpoint_path", "checkpoint_size", "checkpoint_mtime_ns"}
            if not required.issubset(z.files):
                return False
            stat = checkpoint_path_.stat()
            return (
                str(scalar_from_npz(z, "checkpoint_path")) == str(checkpoint_path_)
                and int(scalar_from_npz(z, "checkpoint_size")) == stat.st_size
                and int(scalar_from_npz(z, "checkpoint_mtime_ns")) == stat.st_mtime_ns
            )
    except Exception:
        return False


@torch.inference_mode()
def extract_one(
    task: str,
    method: str,
    seed: int,
    ckpt_path: Path,
    raw_x: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: np.ndarray,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    npz_path = output_dir / "npz" / f"{task}_{method}_seed{seed}_target_predictions.npz"
    if not overwrite and cached_npz_matches(npz_path, ckpt_path):
        with np.load(npz_path, allow_pickle=False) as z:
            return {
                "labels": z["target_labels"].astype(np.int64),
                "predictions": z["target_predictions"].astype(np.int64),
                "probabilities": z["target_probabilities"].astype(np.float64),
                "sample_ids": z["target_sample_ids"].astype(str),
                "threshold": float(scalar_from_npz(z, "threshold")),
                "npz_path": npz_path,
                "checkpoint_path": ckpt_path,
                "cached": True,
            }

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected dictionary checkpoint: {ckpt_path}")
    model, mean, std, threshold, strict_greater = checkpoint_contract(checkpoint, method, task, seed)
    model.to(device).eval()

    dataset = TargetDataset(raw_x, labels, sample_ids, mean, std, apply_dann_log1p=(method == "dann"))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_ids: list[str] = []
    for inputs, batch_labels, batch_ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        # FT_extract evaluated with CUDA AMP; the three DA trainers evaluated
        # in float32.  Reuse those contracts so threshold-edge predictions
        # match the original results as closely as the selected device allows.
        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and method == "full_ft"),
        ):
            output = model(inputs)
        logits = output[0] if isinstance(output, tuple) else output
        all_probs.append(torch.sigmoid(logits).view(-1).float().cpu().numpy())
        all_labels.append(batch_labels.view(-1).cpu().numpy().astype(np.int64))
        all_ids.extend(str(value) for value in batch_ids)

    target_labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probs).astype(np.float64)
    predictions = (probabilities > threshold if strict_greater else probabilities >= threshold).astype(np.int64)
    extracted_ids = np.asarray(all_ids, dtype=str)
    if not np.array_equal(extracted_ids, sample_ids) or not np.array_equal(target_labels, labels.numpy()):
        raise RuntimeError(f"Inference loader changed target sample order/labels: {task}/{method}/seed{seed}")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    stat = ckpt_path.stat()
    np.savez_compressed(
        npz_path,
        target_labels=target_labels,
        target_predictions=predictions,
        target_probabilities=probabilities,
        target_sample_ids=extracted_ids,
        task=task,
        method=method,
        seed=seed,
        threshold=threshold,
        comparison=(">" if strict_greater else ">="),
        normalization_mean=mean,
        normalization_std=std,
        checkpoint_path=str(ckpt_path),
        checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns,
    )

    model.to("cpu")
    del model, checkpoint, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "labels": target_labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "sample_ids": extracted_ids,
        "threshold": threshold,
        "npz_path": npz_path,
        "checkpoint_path": ckpt_path,
        "cached": False,
    }


def verify_and_concatenate(
    task: str,
    method: str,
    seeds: list[int],
    seed_results: Mapping[int, Mapping[str, Any]],
    global_ids: np.ndarray,
    global_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    reference = seed_results[seeds[0]]
    rows: list[dict[str, Any]] = []
    labels_to_concat: list[np.ndarray] = []
    preds_to_concat: list[np.ndarray] = []
    for seed in seeds:
        result = seed_results[seed]
        ids_same_seed0 = np.array_equal(result["sample_ids"], reference["sample_ids"])
        labels_same_seed0 = np.array_equal(result["labels"], reference["labels"])
        ids_same_dataset = np.array_equal(result["sample_ids"], global_ids)
        labels_same_dataset = np.array_equal(result["labels"], global_labels)
        if not (ids_same_seed0 and labels_same_seed0 and ids_same_dataset and labels_same_dataset):
            raise ValueError(
                f"Target sample order/label mismatch: {task}/{method}/seed{seed}; "
                f"ids_vs_seed0={ids_same_seed0}, labels_vs_seed0={labels_same_seed0}, "
                f"ids_vs_dataset={ids_same_dataset}, labels_vs_dataset={labels_same_dataset}"
            )
        labels_to_concat.append(np.asarray(result["labels"], dtype=np.int64))
        preds_to_concat.append(np.asarray(result["predictions"], dtype=np.int64))
        rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "target_test_n": len(result["labels"]),
                "sample_order_matches_seed0": ids_same_seed0,
                "labels_match_seed0": labels_same_seed0,
                "sample_order_matches_dataset": ids_same_dataset,
                "labels_match_dataset": labels_same_dataset,
            }
        )
    return np.concatenate(labels_to_concat), np.concatenate(preds_to_concat), rows


def calculate_and_save_metrics(
    task: str, method: str, labels: np.ndarray, predictions: np.ndarray, output_dir: Path
) -> dict[str, Any]:
    counts = confusion_matrix(labels, predictions, labels=CLASS_IDS)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts, row_sums, out=np.zeros_like(counts, dtype=np.float64), where=row_sums != 0
    )
    pd.DataFrame(counts, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / f"{task}_confusion_matrix_{method}_counts.csv"
    )
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / f"{task}_confusion_matrix_{method}_normalized.csv"
    )
    return {
        "task": task,
        "task_label": TASKS[task]["label"],
        "method": method,
        "method_label": METHOD_LABELS[method],
        "n_predictions": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASS_IDS, average="macro", zero_division=0)),
        "earthquake_recall": float(normalized[0, 0]),
        "explosion_recall": float(normalized[1, 1]),
        "confusion_matrix": json.dumps(counts.tolist()),
    }


def plot_task(task: str, metrics: Mapping[str, Mapping[str, Any]], output_dir: Path, num_seeds: int) -> None:
    # Reset pyplot state so the second task has exactly the same layout as the
    # first when PNG/PDF/SVG figures are produced sequentially.
    plt.close("all")
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=False)
    fig.subplots_adjust(
        left=0.135,
        right=0.835,
        bottom=0.105,
        top=0.875,
        wspace=0.22,
        hspace=0.30,
    )
    image = None
    for ax, method in zip(axes.ravel(), METHODS):
        row = metrics[method]
        counts = np.asarray(json.loads(str(row["confusion_matrix"])), dtype=np.int64)
        row_sums = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts, row_sums, out=np.zeros_like(counts, dtype=np.float64), where=row_sums != 0
        )
        image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"({PANEL_LETTERS[method]}) {METHOD_LABELS[method]} "
            f"(Macro-F1 = {float(row['macro_f1']):.3f})",
            fontsize=13,
        )
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                value = normalized[i, j]
                ax.text(
                    j,
                    i,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.5 else "black",
                    fontsize=11,
                )
    fig.suptitle(
        TASKS[task]["label"],
        fontsize=17,
        y=0.970,
    )
    fig.supxlabel("Predicted Class", fontsize=15, y=0.018)
    fig.supylabel("True Class", fontsize=15, x=0.025)
    if image is not None:
        colorbar_axis = fig.add_axes([0.865, 0.155, 0.022, 0.665])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Row-normalized proportion", fontsize=12)
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{task}_confusion_matrices_multiseed.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be >= 1 and num-workers must be >= 0")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be a non-empty list without duplicates")

    checkpoints = preflight(args.tasks, args.seeds)
    print(f"[PREFLIGHT] data: {ENAM_TEST_PT}", flush=True)
    print(f"[PREFLIGHT] checkpoints: {len(checkpoints)} complete", flush=True)
    if args.preflight_only:
        print("[DONE] Preflight passed.", flush=True)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[DEVICE] {device}", flush=True)
    raw_x, target_labels_tensor, target_ids = load_target_test()
    target_labels = target_labels_tensor.numpy().astype(np.int64)
    print(f"[DATA] ENAM target-test samples: {len(target_labels)}", flush=True)

    performance_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for task in args.tasks:
        task_metrics: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            seed_results: dict[int, dict[str, Any]] = {}
            for seed in args.seeds:
                ckpt_path = checkpoints[(task, method, seed)]
                print(f"[INFER] {task} | {method} | seed={seed} | {ckpt_path}", flush=True)
                result = extract_one(
                    task,
                    method,
                    seed,
                    ckpt_path,
                    raw_x,
                    target_labels_tensor,
                    target_ids,
                    output_dir,
                    device,
                    args.batch_size,
                    args.num_workers,
                    args.overwrite_npz,
                )
                seed_results[seed] = result
                checkpoint_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "checkpoint_path": str(ckpt_path),
                        "npz_path": str(result["npz_path"]),
                        "threshold": result["threshold"],
                        "used_cached_npz": result["cached"],
                    }
                )

            labels_concat, predictions_concat, rows = verify_and_concatenate(
                task, method, args.seeds, seed_results, target_ids, target_labels
            )
            consistency_rows.extend(rows)
            metric_row = calculate_and_save_metrics(
                task, method, labels_concat, predictions_concat, output_dir
            )
            metric_row["num_seeds"] = len(args.seeds)
            metric_row["target_test_n_per_seed"] = len(target_labels)
            task_metrics[method] = metric_row
            performance_rows.append(metric_row)
            print(
                f"[METRIC] {TASKS[task]['label']} | {METHOD_LABELS[method]} | "
                f"n={len(labels_concat)} | accuracy={metric_row['accuracy']:.6f} | "
                f"macro_f1={metric_row['macro_f1']:.6f}",
                flush=True,
            )
        plot_task(task, task_metrics, output_dir, len(args.seeds))

    pd.DataFrame(performance_rows).to_csv(
        output_dir / "performance_multiseed_from_concatenated_predictions.csv", index=False
    )
    pd.DataFrame(consistency_rows).to_csv(
        output_dir / "target_test_order_label_consistency_multiseed.csv", index=False
    )
    pd.DataFrame(checkpoint_rows).to_csv(
        output_dir / "used_checkpoints_multiseed_r010.csv", index=False
    )
    print(f"[DONE] Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
