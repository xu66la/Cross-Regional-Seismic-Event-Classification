#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract wide CSVs and plot DANN ratio trends for 12 transfer directions.

Input is the row-oriented JSON produced by ``summarize_dann_label_ratios.py``.
The script requires complete 0/1/5/10/20% x 12 directions x 5 seeds results,
then creates one Accuracy CSV/figure and one Macro-F1 CSV/figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "DANN_target_label_ratio"
    / "dann_label_ratio_target_metrics_5seed.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "target_label_ratio_analysis"
)

RATIOS = (0, 1, 5, 10, 20)
DOMAINS = ("base", "msh", "enam", "hlp")
DOMAIN_LABELS = {
    "base": "BASE",
    "msh": "MSH",
    "enam": "ENAM",
    "hlp": "HLP",
}
SOURCE_COLORS = {
    "base": "#1f77b4",
    "msh": "#ff7f0e",
    "enam": "#2ca02c",
    "hlp": "#d62728",
}
TARGET_MARKERS = {
    "base": "D",
    "msh": "o",
    "enam": "s",
    "hlp": "^",
}
METRICS = {
    "accuracy": {
        "json_key": "accuracy_mean",
        "title": "DANN Accuracy for 12 Transfer Directions",
        "ylabel": "Accuracy",
        "ylim": (0.40, 0.96),
        "stem": "dann_accuracy_12_transfer_directions_labeled_target_lines",
    },
    "macro_f1": {
        "json_key": "macro_f1_mean",
        "title": "DANN Macro-F1 for 12 Transfer Directions",
        "ylabel": "Macro-F1",
        "ylim": (0.34, 0.93),
        "stem": "dann_macro_f1_12_transfer_directions_labeled_target_lines",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot 0/1/5/10/20% DANN trends for 12 transfer directions"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def transfer_directions() -> list[str]:
    return [
        f"{source}->{target}"
        for source in DOMAINS
        for target in DOMAINS
        if source != target
    ]


def display_pair(pair: str) -> str:
    source, target = pair.split("->")
    return f"{DOMAIN_LABELS[source]}->{DOMAIN_LABELS[target]}"


def load_complete_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a rows list")

    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    errors: list[str] = []
    expected_pairs = transfer_directions()
    for row in rows:
        try:
            ratio = int(row["label_percent"])
            pair = str(row["transfer"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row in {path}: {row}") from exc
        key = (ratio, pair)
        if key in indexed:
            errors.append(f"duplicate row: {ratio}% {pair}")
        indexed[key] = row

    for ratio in RATIOS:
        for pair in expected_pairs:
            row = indexed.get((ratio, pair))
            if row is None:
                errors.append(f"missing row: {ratio}% {pair}")
                continue
            if int(row.get("n", 0)) != 5:
                errors.append(f"incomplete: {ratio}% {pair}, n={row.get('n', 0)}")
                continue
            for metric in METRICS.values():
                value = row.get(metric["json_key"])
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    errors.append(f"invalid {metric['json_key']}: {ratio}% {pair}")

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        remainder = len(errors) - min(20, len(errors))
        extra = f"\n  ... 另外还有 {remainder} 项" if remainder else ""
        raise ValueError(
            "输入 JSON 尚未包含完整的 5 个比例 × 12 个方向 × 5 个 seed。\n"
            f"{preview}{extra}\n"
            "请等待训练完成，并重新运行 summarize_dann_label_ratios.py。"
        )
    return indexed


def extract_series(
    indexed: dict[tuple[int, str], dict[str, Any]], metric_key: str
) -> dict[str, list[float]]:
    json_key = METRICS[metric_key]["json_key"]
    return {
        pair: [float(indexed[(ratio, pair)][json_key]) for ratio in RATIOS]
        for pair in transfer_directions()
    }


def save_wide_csv(
    path: Path, series: dict[str, list[float]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", *[f"{ratio}%" for ratio in RATIOS]])
        for pair in transfer_directions():
            writer.writerow(
                [display_pair(pair), *[f"{value:.6f}" for value in series[pair]]]
            )


def plot_metric(
    metric_key: str,
    series: dict[str, list[float]],
    output_dir: Path,
) -> list[Path]:
    config = METRICS[metric_key]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.asarray(RATIOS, dtype=float)

    all_values = []
    for pair in transfer_directions():
        source, target = pair.split("->")
        values = series[pair]
        all_values.append(values)
        ax.plot(
            x,
            values,
            color=SOURCE_COLORS[source],
            marker=TARGET_MARKERS[target],
            markersize=4.4,
            linewidth=2.0,
            label=display_pair(pair),
        )

    mean_values = np.asarray(all_values, dtype=float).mean(axis=0)
    ax.plot(
        x,
        mean_values,
        color="#111111",
        marker="o",
        markersize=4.4,
        linewidth=2.0,
        label="Mean",
        zorder=8,
    )

    ax.set_title(config["title"], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Labeled target-domain data ratio (%)", fontsize=12)
    ax.set_ylabel(config["ylabel"], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ratio) for ratio in RATIOS])
    ax.set_xlim(-0.8, 20.8)
    ax.set_ylim(*config["ylim"])
    ax.tick_params(axis="both", labelsize=9.5, length=3.5, width=0.8)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.55, color="#D8D8D8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend = ax.legend(
        title="Transfer direction",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=8.0,
        title_fontsize=9.0,
        borderpad=0.95,
        labelspacing=0.95,
        handlelength=2.1,
    )
    legend.get_frame().set_edgecolor("#C8C8C8")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)

    fig.tight_layout()
    stem = config["stem"]
    outputs = [output_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indexed = load_complete_rows(input_path)
    outputs: list[Path] = []
    for metric_key in ("accuracy", "macro_f1"):
        series = extract_series(indexed, metric_key)
        csv_path = output_dir / f"{METRICS[metric_key]['stem']}.csv"
        save_wide_csv(csv_path, series)
        outputs.append(csv_path)
        outputs.extend(plot_metric(metric_key, series, output_dir))

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
