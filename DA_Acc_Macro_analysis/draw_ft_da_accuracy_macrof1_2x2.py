#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw separate 2x2 figures for target-domain Accuracy and Macro-F1.

Panels are (a) FT baseline, (b) DAN, (c) DSAN, and (d) DANN.  Each panel is
a 4x4 source/target matrix whose diagonal is intentionally empty.  Adaptation
panels also show their delta relative to FT for the same transfer direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    "./outputs/da_10pct_target_metrics_5seed.csv"
)
DEFAULT_OUTPUT_DIR = Path("./outputs")

# Keep the domain order used by the reference figure.
DOMAINS = ["BASE", "MSH", "ENAM", "HLP"]
DOMAIN_KEYS = [domain.lower() for domain in DOMAINS]
METHODS = ["FT", "DAN", "DSAN", "DANN"]
PANEL_LABELS = {"FT": "(a)", "DAN": "(b)", "DSAN": "(c)", "DANN": "(d)"}
METHOD_TITLES = {
    "FT": "Full fine-tuning (Baseline)",
    "DAN": "DAN",
    "DSAN": "DSAN",
    "DANN": "DANN",
}
METRICS = {
    "accuracy": {
        "column": "accuracy_mean",
        "label": "Accuracy",
        "stem": "Accuracy_FT_DA_comparison_2x2",
    },
    "macro_f1": {
        "column": "macro_f1_mean",
        "label": "Macro-F1",
        "stem": "MacroF1_FT_DA_comparison_2x2",
    },
}

COLORS = {
    "baseline": "#FFFFFF",
    "diag": "#F2F2F2",
    "improve": "#DCEAF4",
    "similar": "#FFF2CC",
    "decline": "#F4D6C6",
    "missing": "#E6E6E6",
}

INNER_GRID_COLOR = "#777777"
OUTER_GRID_COLOR = "#777777"
INNER_GRID_WIDTH = 0.55
OUTER_GRID_WIDTH = 0.8

FIG_WIDTH_IN = 7.5
FIG_HEIGHT_IN = 7.0
BASELINE_VALUE_FONTSIZE = 9.5
METHOD_VALUE_FONTSIZE = 9.2
DELTA_FONTSIZE = 8.0
TITLE_FONTSIZE = 10.8
SHARED_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 10.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw FT/DAN/DSAN/DANN 2x2 Accuracy and Macro-F1 figures"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow missing or n<5 results and render those cells as N/A",
    )
    return parser.parse_args()


def choose_font_family() -> str:
    for family in ("Arial", "Times New Roman", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


def load_results(csv_path: Path, allow_incomplete: bool) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = {
        "method", "source", "target", "n", "missing_seeds",
        "accuracy_mean", "macro_f1_mean",
    }
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing_columns)}")

    data["method"] = data["method"].astype(str).str.upper()
    data["source"] = data["source"].astype(str).str.lower()
    data["target"] = data["target"].astype(str).str.lower()
    data = data[data["method"].isin(METHODS)].copy()

    problems = []
    for method in METHODS:
        for source in DOMAIN_KEYS:
            for target in DOMAIN_KEYS:
                if source == target:
                    continue
                rows = data[
                    (data["method"] == method)
                    & (data["source"] == source)
                    & (data["target"] == target)
                ]
                if len(rows) != 1:
                    problems.append(f"{method} {source}->{target}: rows={len(rows)}")
                    continue
                row = rows.iloc[0]
                if int(row["n"]) != 5:
                    problems.append(f"{method} {source}->{target}: n={int(row['n'])}")
                    continue
                for column in ("accuracy_mean", "macro_f1_mean"):
                    if pd.isna(row[column]):
                        problems.append(f"{method} {source}->{target}: {column}=NaN")

    if problems and not allow_incomplete:
        preview = "\n".join(f"  - {item}" for item in problems[:20])
        remainder = len(problems) - min(20, len(problems))
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise ValueError(
            "CSV is not a complete 4 methods x 12 transfers x 5 seeds result.\n"
            f"{preview}{suffix}\n"
            "Wait for all runs and rebuild the CSV, or use --allow-incomplete for a draft."
        )
    if problems:
        print(f"Warning: drawing an incomplete draft ({len(problems)} incomplete cells).")
    return data


def load_matrix(data: pd.DataFrame, method: str, column: str) -> np.ndarray:
    matrix = np.full((len(DOMAINS), len(DOMAINS)), np.nan, dtype=float)
    for row_index, source in enumerate(DOMAIN_KEYS):
        for column_index, target in enumerate(DOMAIN_KEYS):
            if source == target:
                continue
            rows = data[
                (data["method"] == method)
                & (data["source"] == source)
                & (data["target"] == target)
            ]
            if len(rows) == 1 and int(rows.iloc[0]["n"]) == 5:
                value = rows.iloc[0][column]
                if pd.notna(value):
                    matrix[row_index, column_index] = float(value)
    return matrix


def delta_color(delta: float) -> str:
    if delta > 0.02:
        return COLORS["improve"]
    if delta < -0.02:
        return COLORS["decline"]
    return COLORS["similar"]


def format_signed(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.3f}"


def setup_axis(ax, show_xlabels: bool, show_ylabels: bool) -> None:
    ax.set_xlim(-0.5, len(DOMAINS) - 0.5)
    ax.set_ylim(len(DOMAINS) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(DOMAINS)))
    ax.set_yticks(np.arange(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS if show_xlabels else [], fontsize=9)
    ax.set_yticklabels(DOMAINS if show_ylabels else [], fontsize=9)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_outer_border(ax) -> None:
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), len(DOMAINS), len(DOMAINS),
            facecolor="none", edgecolor=OUTER_GRID_COLOR,
            linewidth=OUTER_GRID_WIDTH, zorder=5,
        )
    )


def draw_cell(ax, row: int, col: int, facecolor: str) -> None:
    ax.add_patch(
        Rectangle(
            (col - 0.5, row - 0.5), 1, 1,
            facecolor=facecolor, edgecolor=INNER_GRID_COLOR,
            linewidth=INNER_GRID_WIDTH,
        )
    )


def draw_baseline_matrix(
    ax, matrix: np.ndarray, show_xlabels: bool, show_ylabels: bool
) -> None:
    setup_axis(ax, show_xlabels, show_ylabels)
    for i in range(len(DOMAINS)):
        for j in range(len(DOMAINS)):
            if i == j:
                draw_cell(ax, i, j, COLORS["diag"])
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#A8A8A8")
            elif np.isnan(matrix[i, j]):
                draw_cell(ax, i, j, COLORS["missing"])
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.5, color="#777777")
            else:
                draw_cell(ax, i, j, COLORS["baseline"])
                ax.text(
                    j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=BASELINE_VALUE_FONTSIZE, color="#111111",
                )
    draw_outer_border(ax)


def draw_method_matrix(
    ax,
    method_matrix: np.ndarray,
    baseline_matrix: np.ndarray,
    show_xlabels: bool,
    show_ylabels: bool,
) -> None:
    setup_axis(ax, show_xlabels, show_ylabels)
    for i in range(len(DOMAINS)):
        for j in range(len(DOMAINS)):
            if i == j:
                draw_cell(ax, i, j, COLORS["diag"])
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#A8A8A8")
                continue
            value = method_matrix[i, j]
            baseline = baseline_matrix[i, j]
            if np.isnan(value) or np.isnan(baseline):
                draw_cell(ax, i, j, COLORS["missing"])
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.5, color="#777777")
                continue
            delta = value - baseline
            draw_cell(ax, i, j, delta_color(delta))
            ax.text(
                j, i - 0.10, f"{value:.3f}", ha="center", va="center",
                fontsize=METHOD_VALUE_FONTSIZE, color="#111111",
            )
            ax.text(
                j, i + 0.18, f"Δ {format_signed(delta)}", ha="center", va="center",
                fontsize=DELTA_FONTSIZE, color="#333333",
            )
    draw_outer_border(ax)


def mean_delta(method: np.ndarray, baseline: np.ndarray) -> float:
    mask = np.isfinite(method) & np.isfinite(baseline)
    np.fill_diagonal(mask, False)
    return float(np.mean(method[mask] - baseline[mask])) if mask.any() else float("nan")


def draw_metric_figure(
    metric_key: str, data: pd.DataFrame, output_dir: Path
) -> list[Path]:
    config = METRICS[metric_key]
    matrices = {
        method: load_matrix(data, method, config["column"])
        for method in METHODS
    }
    baseline = matrices["FT"]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=600)
    order = [["FT", "DAN"], ["DSAN", "DANN"]]

    for row_index in range(2):
        for column_index in range(2):
            ax = axes[row_index, column_index]
            method = order[row_index][column_index]
            show_xlabels = row_index == 1
            show_ylabels = column_index == 0
            if method == "FT":
                draw_baseline_matrix(ax, matrices[method], show_xlabels, show_ylabels)
                title = f"{PANEL_LABELS[method]} {METHOD_TITLES[method]}"
            else:
                draw_method_matrix(
                    ax, matrices[method], baseline, show_xlabels, show_ylabels
                )
                delta = mean_delta(matrices[method], baseline)
                delta_text = "N/A" if np.isnan(delta) else format_signed(delta)
                title = (
                    f"{PANEL_LABELS[method]} {METHOD_TITLES[method]} "
                    f"(Mean Δ = {delta_text})"
                )
            ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="semibold", pad=7)

    for ax in axes[:, 0]:
        ax.set_anchor("E")
    for ax in axes[:, 1]:
        ax.set_anchor("W")

    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.925, bottom=0.175,
        wspace=0.045, hspace=0.16,
    )
    fig.canvas.draw()
    left_top = axes[0, 0].get_position()
    right_top = axes[0, 1].get_position()
    left_bottom = axes[1, 0].get_position()
    right_bottom = axes[1, 1].get_position()
    grid_x0 = left_top.x0
    grid_x1 = right_top.x1
    grid_y0 = min(left_bottom.y0, right_bottom.y0)
    grid_y1 = max(left_top.y1, right_top.y1)

    fig.text(
        grid_x0 - 0.075, (grid_y0 + grid_y1) / 2, "Source domain",
        ha="center", va="center", rotation=90, fontsize=SHARED_LABEL_FONTSIZE,
    )
    fig.text(
        (grid_x0 + grid_x1) / 2, 0.114, "Target domain",
        ha="center", va="center", fontsize=SHARED_LABEL_FONTSIZE,
    )

    legend_handles = [
        Patch(facecolor=COLORS["improve"], edgecolor="#777777", linewidth=0.6, label="Better: Δ > 0.02"),
        Patch(facecolor=COLORS["decline"], edgecolor="#777777", linewidth=0.6, label="Worse: Δ < −0.02"),
        Patch(facecolor=COLORS["similar"], edgecolor="#777777", linewidth=0.6, label="Comparable: |Δ| ≤ 0.02"),
        Patch(facecolor=COLORS["diag"], edgecolor="#777777", linewidth=0.6, label="Same domain"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2, frameon=False,
        fontsize=LEGEND_FONTSIZE, handlelength=1.2, handletextpad=0.45,
        columnspacing=1.45, bbox_to_anchor=((grid_x0 + grid_x1) / 2, 0.030),
        bbox_transform=fig.transFigure,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{config['stem']}.{suffix}" for suffix in ("png", "pdf", "svg")]
    fig.savefig(outputs[0], dpi=600)
    fig.savefig(outputs[1])
    fig.savefig(outputs[2])
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": choose_font_family(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    data = load_results(args.csv.resolve(), args.allow_incomplete)
    outputs = []
    for metric_key in ("accuracy", "macro_f1"):
        outputs.extend(draw_metric_figure(metric_key, data, args.output_dir.resolve()))
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
