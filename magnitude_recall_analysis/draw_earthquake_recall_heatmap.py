#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw 4x4 domain-transfer earthquake-recall heatmaps by magnitude.

Expected input (one row per transfer direction and magnitude bin):

    task,magnitude_bin,full_ft,dan,dann,dsan
    base_to_msh,0-1,...

Recall values must be ratios in [0, 1].  NaN is allowed when the target test
set has no earthquake samples in a magnitude bin; such cells are drawn as an
em dash.  Additional columns in the CSV are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "magnitude_recall_analysis"
)

DEFAULT_INPUT = BASE_DIR / "gain_connect.csv"

DEFAULT_OUTPUT_DIR = BASE_DIR


DOMAINS = ("BASE", "MSH", "ENAM", "HLP")
MAGNITUDE_BINS = ("0-1", "1-2", "2-3", ">=3")
METHODS = {
    "full_ft": "Full Fine-tuning",
    "dan": "DAN",
    "dann": "DANN",
    "dsan": "DSAN",
}
REQUIRED_COLUMNS = {"task", "magnitude_bin", *METHODS}
EXPECTED_TASKS = {
    f"{source.lower()}_to_{target.lower()}"
    for source in DOMAINS
    for target in DOMAINS
    if source != target
}

CMAP = LinearSegmentedColormap.from_list(
    "recall_blues", ["#FFFFFF", "#DCEAF4", "#8FB3CC", "#2C5E87"]
)
EMPTY_COLOR = "#F2F2F2"
GRID_COLOR = "#777777"
OUTER_GRID_COLOR = "#555555"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw earthquake-recall heatmaps for four methods and four magnitude bins."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stem", default="fullft_dan_dann_dsan_earthquake_recall_heatmap"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow missing task/bin rows (shown as em dashes).",
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


def normalize_task(value: object) -> str:
    return str(value).strip().lower().replace("->", "_to_")


def load_and_validate(path: Path, allow_incomplete: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    df = df.copy()
    df["task"] = df["task"].map(normalize_task)
    df["magnitude_bin"] = df["magnitude_bin"].astype(str).str.strip()
    invalid_tasks = sorted(set(df["task"]) - EXPECTED_TASKS)
    invalid_bins = sorted(set(df["magnitude_bin"]) - set(MAGNITUDE_BINS))
    if invalid_tasks:
        raise ValueError(f"Invalid transfer tasks: {invalid_tasks}")
    if invalid_bins:
        raise ValueError(f"Invalid magnitude bins: {invalid_bins}")
    duplicated = df.duplicated(["task", "magnitude_bin"], keep=False)
    if duplicated.any():
        pairs = df.loc[duplicated, ["task", "magnitude_bin"]].drop_duplicates()
        raise ValueError(f"Duplicate task/bin rows:\n{pairs.to_string(index=False)}")

    for method in METHODS:
        df[method] = pd.to_numeric(df[method], errors="coerce")
        invalid = df[method].notna() & ~df[method].between(0.0, 1.0)
        if invalid.any():
            raise ValueError(f"{method} contains recall values outside [0, 1]")

    expected_keys = {
        (task, magnitude_bin)
        for task in EXPECTED_TASKS
        for magnitude_bin in MAGNITUDE_BINS
    }
    present_keys = set(zip(df["task"], df["magnitude_bin"]))
    missing_keys = sorted(expected_keys - present_keys)
    if missing_keys and not allow_incomplete:
        preview = ", ".join(f"{task}/{bin_}" for task, bin_ in missing_keys[:12])
        suffix = f" ... and {len(missing_keys) - 12} more" if len(missing_keys) > 12 else ""
        raise ValueError(
            f"Input is missing {len(missing_keys)} of the expected 48 task/bin rows: "
            f"{preview}{suffix}. Use --allow-incomplete only for drafts."
        )
    return df


def task_to_domains(task: str) -> tuple[str, str]:
    source, target = task.split("_to_")
    return source.upper(), target.upper()


def build_method_matrix(df: pd.DataFrame, method: str, magnitude_bin: str) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=DOMAINS, columns=DOMAINS, dtype=float)
    for _, row in df[df["magnitude_bin"] == magnitude_bin].iterrows():
        source, target = task_to_domains(row["task"])
        matrix.loc[source, target] = row[method]
    return matrix


def setup_axis(ax, show_xlabels: bool, show_ylabels: bool) -> None:
    ax.set_xlim(-0.5, len(DOMAINS) - 0.5)
    ax.set_ylim(len(DOMAINS) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(DOMAINS)))
    ax.set_yticks(np.arange(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS if show_xlabels else [], fontsize=8.6)
    ax.set_yticklabels(DOMAINS if show_ylabels else [], fontsize=8.6)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_matrix(
    ax,
    matrix: pd.DataFrame,
    title: str,
    norm: Normalize,
    show_xlabels: bool,
    show_ylabels: bool,
) -> None:
    setup_axis(ax, show_xlabels, show_ylabels)
    for row_index, source in enumerate(DOMAINS):
        for column_index, target in enumerate(DOMAINS):
            value = matrix.loc[source, target]
            empty = source == target or pd.isna(value)
            ax.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    facecolor=EMPTY_COLOR if empty else CMAP(norm(value)),
                    edgecolor=GRID_COLOR,
                    linewidth=0.6,
                )
            )
            if empty:
                ax.text(
                    column_index,
                    row_index,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="#A8A8A8",
                )
            else:
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#1A1A1A",
                )
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            len(DOMAINS),
            len(DOMAINS),
            facecolor="none",
            edgecolor=OUTER_GRID_COLOR,
            linewidth=1.0,
            zorder=5,
        )
    )
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=7)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(input_path, args.allow_incomplete)

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": choose_font_family(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    matrices = {
        magnitude_bin: {
            method: build_method_matrix(df, method, magnitude_bin) for method in METHODS
        }
        for magnitude_bin in MAGNITUDE_BINS
    }
    finite_values = np.concatenate(
        [
            matrix.to_numpy(dtype=float)[np.isfinite(matrix.to_numpy(dtype=float))]
            for by_method in matrices.values()
            for matrix in by_method.values()
        ]
    )
    if finite_values.size == 0:
        raise ValueError("Input contains no finite recall values")
    value_min, value_max = float(finite_values.min()), float(finite_values.max())
    if np.isclose(value_min, value_max):
        value_min, value_max = 0.0, 1.0
    norm = Normalize(vmin=value_min, vmax=value_max)

    fig, axes = plt.subplots(len(MAGNITUDE_BINS), len(METHODS), figsize=(11.6, 11.2), dpi=300)
    for row_index, magnitude_bin in enumerate(MAGNITUDE_BINS):
        for column_index, method in enumerate(METHODS):
            draw_matrix(
                axes[row_index, column_index],
                matrices[magnitude_bin][method],
                METHODS[method] if row_index == 0 else "",
                norm,
                show_xlabels=row_index == len(MAGNITUDE_BINS) - 1,
                show_ylabels=column_index == 0,
            )

    fig.subplots_adjust(left=0.09, right=0.90, top=0.93, bottom=0.08, wspace=0.005, hspace=0.06)
    for row_index, magnitude_bin in enumerate(MAGNITUDE_BINS):
        position = axes[row_index, 0].get_position()
        fig.text(
            0.040,
            (position.y0 + position.y1) / 2,
            f"Magnitude {magnitude_bin}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=10.5,
            fontweight="semibold",
        )
    fig.text(0.018, 0.505, "Source domain", ha="center", va="center", rotation=90, fontsize=12)
    fig.text(0.50, 0.035, "Target domain", ha="center", va="center", fontsize=12)
    colorbar_axis = fig.add_axes([0.925, 0.12, 0.014, 0.76])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=colorbar_axis)
    colorbar.set_label("Earthquake recall", fontsize=10)
    colorbar.ax.tick_params(labelsize=9, length=3, width=0.7)
    colorbar.outline.set_linewidth(0.7)

    outputs = [output_dir / f"{args.stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    fig.savefig(outputs[2])
    plt.close(fig)
    for output in outputs:
        print(f"[OK] saved: {output}")


if __name__ == "__main__":
    main()
