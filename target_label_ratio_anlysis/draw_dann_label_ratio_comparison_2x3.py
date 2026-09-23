#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw DANN labeled-target-ratio comparison matrices.

The script creates two figures:
1. Accuracy comparison across 0%, 1%, 5%, 10%, and 20% labeled target data.
2. Macro-F1 comparison across the same labeled-target ratios.

The 0% figure is the UDA reference. Other panels show SSDA values and their
delta relative to the corresponding UDA source-target transfer pair.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = SCRIPT_DIR
DATA_DIR =  PROJECT_ROOT / "outputs"/ "DANN_target_label_ratio"

UDA_JSON = DATA_DIR / "dann_stft_uda_0pct_summary.json"
LABELED_TARGET_RATIOS = [1, 5, 10, 20]
SSDA_JSONS = {
    ratio: DATA_DIR / f"dann_stft_ssda_{ratio}pct_summary.json"
    for ratio in LABELED_TARGET_RATIOS
}

OUTPUTS = {
    "accuracy": {
        "display_name": "Accuracy",
        "png": BASE_DIR / "dann_labeled_target_accuracy_comparison_2x3.png",
        "pdf": BASE_DIR / "dann_labeled_target_accuracy_comparison_2x3.pdf",
        "svg": BASE_DIR / "dann_labeled_target_accuracy_comparison_2x3.svg",
    },
    "macro_f1": {
        "display_name": "Macro-F1",
        "png": BASE_DIR / "dann_labeled_target_macro_f1_comparison_2x3.png",
        "pdf": BASE_DIR / "dann_labeled_target_macro_f1_comparison_2x3.pdf",
        "svg": BASE_DIR / "dann_labeled_target_macro_f1_comparison_2x3.svg",
    },
}

DOMAINS = ["BASE", "MSH", "ENAM", "HLP"]
DOMAIN_KEYS = [domain.lower() for domain in DOMAINS]
PANEL_ORDER = [(0, (0, 0)), (1, (0, 1)), (5, (0, 2)), (10, (1, 1)), (20, (1, 2))]
RATIOS = [0, *LABELED_TARGET_RATIOS]

COLORS = {
    "uda": "#FFFFFF",
    "diag": "#F2F2F2",
}
LINE_COLORS = {
    "accuracy": "#1F4E79",
    "macro_f1": "#4F7F7A",
}
DELTA_CMAP = LinearSegmentedColormap.from_list("delta_blues", ["#FFFFFF", "#DCEAF4", "#1F4E79"])

INNER_GRID_COLOR = "#777777"
OUTER_GRID_COLOR = "#777777"
INNER_GRID_WIDTH = 0.55
OUTER_GRID_WIDTH = 0.8

FIG_WIDTH_IN = 12.0
FIG_HEIGHT_IN = 7.8
LEFT_COLUMN_SHIFT_CM = 1.5
PANEL_LABEL_SHIFT_LEFT_CM = 1.2
VALUE_FONTSIZE = 8.8
DELTA_FONTSIZE = 7.2
TITLE_FONTSIZE = 10.7
SHARED_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 10.0
PANEL_LABEL_FONTSIZE = 11


def choose_font_family() -> str:
    for family in ("Arial", "Times New Roman", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


def round_half_up(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def format_signed(value: float | Decimal) -> str:
    decimal_value = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    sign = "+" if decimal_value >= 0 else "−"
    return f"{sign}{abs(decimal_value):.3f}"


def load_metric_matrix(path: Path, metric: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)["summary"]

    matrix = pd.DataFrame(np.nan, index=DOMAINS, columns=DOMAINS, dtype=float)
    for source_key, source in zip(DOMAIN_KEYS, DOMAINS):
        for target_key, target in zip(DOMAIN_KEYS, DOMAINS):
            if source_key == target_key:
                continue
            pair_key = f"{source_key}->{target_key}"
            matrix.loc[source, target] = summary[pair_key]["aggregate"]["tgt_test"][metric]["mean"]
    return matrix


def offdiag_mean_delta(matrix: pd.DataFrame, reference: pd.DataFrame) -> Decimal:
    deltas: list[Decimal] = []
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            deltas.append(round_half_up(matrix.loc[source, target] - reference.loc[source, target]))
    return sum(deltas, Decimal("0")) / Decimal(len(deltas))


def mean_deltas_by_metric() -> dict[str, list[float]]:
    result = {}
    for metric in ("accuracy", "macro_f1"):
        reference = load_metric_matrix(UDA_JSON, metric)
        values = []
        for ratio in RATIOS:
            matrix = reference if ratio == 0 else load_metric_matrix(SSDA_JSONS[ratio], metric)
            values.append(float(offdiag_mean_delta(matrix, reference)))
        result[metric] = values
    return result


def global_delta_norm() -> Normalize:
    maxima = []
    for metric in ("accuracy", "macro_f1"):
        reference = load_metric_matrix(UDA_JSON, metric)
        for ratio, path in SSDA_JSONS.items():
            matrix = load_metric_matrix(path, metric)
            for source in DOMAINS:
                for target in DOMAINS:
                    if source != target:
                        maxima.append(max(0.0, matrix.loc[source, target] - reference.loc[source, target]))
    vmax = max(maxima) if maxima else 1.0
    return Normalize(vmin=0.0, vmax=vmax, clip=True)


def setup_axis(ax, show_xlabels: bool, show_ylabels: bool) -> None:
    ax.set_xlim(-0.5, len(DOMAINS) - 0.5)
    ax.set_ylim(len(DOMAINS) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(DOMAINS)))
    ax.set_yticks(np.arange(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS if show_xlabels else [], fontsize=8.5)
    ax.set_yticklabels(DOMAINS if show_ylabels else [], fontsize=8.5)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_outer_border(ax) -> None:
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            len(DOMAINS),
            len(DOMAINS),
            facecolor="none",
            edgecolor=OUTER_GRID_COLOR,
            linewidth=OUTER_GRID_WIDTH,
            zorder=5,
        )
    )


def draw_same_domain(ax, row: int, col: int) -> None:
    ax.text(col, row, "—", ha="center", va="center", fontsize=12, color="#A8A8A8")


def draw_matrix(
    ax,
    matrix: pd.DataFrame,
    reference: pd.DataFrame,
    ratio: int,
    metric_name: str,
    show_xlabels: bool,
    show_ylabels: bool,
    delta_norm: Normalize,
) -> None:
    setup_axis(ax, show_xlabels, show_ylabels)
    is_reference = ratio == 0

    for i, source in enumerate(DOMAINS):
        for j, target in enumerate(DOMAINS):
            is_diag = source == target
            if is_diag:
                facecolor = COLORS["diag"]
            elif is_reference:
                facecolor = COLORS["uda"]
            else:
                facecolor = DELTA_CMAP(delta_norm(matrix.loc[source, target] - reference.loc[source, target]))

            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor=INNER_GRID_COLOR,
                    linewidth=INNER_GRID_WIDTH,
                )
            )

            if is_diag:
                draw_same_domain(ax, i, j)
                continue

            value = matrix.loc[source, target]
            if is_reference:
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=VALUE_FONTSIZE,
                    color="#111111",
                )
                continue

            delta = value - reference.loc[source, target]
            ax.text(
                j,
                i - 0.10,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=VALUE_FONTSIZE,
                color="#111111",
            )
            ax.text(
                j,
                i + 0.18,
                f"Δ {format_signed(delta)}",
                ha="center",
                va="center",
                fontsize=DELTA_FONTSIZE,
                color="#333333",
            )

    draw_outer_border(ax)
    mean_delta = Decimal("0.000") if is_reference else offdiag_mean_delta(matrix, reference)
    ax.set_title(
        f"{ratio}% (UDA reference)" if is_reference else f"{ratio}% label (Mean Δ = {format_signed(mean_delta)})",
        fontsize=TITLE_FONTSIZE,
        fontweight="semibold",
        pad=7,
    )


def draw_mean_gain_lineplot(ax, metric: str) -> Line2D:
    gains = mean_deltas_by_metric()
    x_positions = np.arange(len(RATIOS))

    line_label = "Accuracy" if metric == "accuracy" else "Macro-F1"
    line = ax.plot(
        x_positions,
        gains[metric],
        color=LINE_COLORS[metric],
        linewidth=2.2,
        marker="o",
        markersize=4.8,
        markerfacecolor=LINE_COLORS[metric],
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=line_label,
    )[0]

    ax.axhline(0, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.2, len(RATIOS) - 0.8)
    ax.set_ylim(-0.015, max(max(gains["accuracy"]), max(gains["macro_f1"])) + 0.030)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{ratio}%" for ratio in RATIOS], fontsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8.5, length=3, width=0.7)
    ax.tick_params(axis="x", length=3, width=0.7, pad=2)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(0.7)

    return line


def add_bottom_legend(fig) -> None:
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["uda"], edgecolor=INNER_GRID_COLOR, linewidth=0.6, label="UDA reference"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["diag"], edgecolor=INNER_GRID_COLOR, linewidth=0.6, label="Same domain"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.020),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        handletextpad=0.55,
        columnspacing=1.55,
    )


def draw_metric_figure(metric: str) -> None:
    output = OUTPUTS[metric]
    reference = load_metric_matrix(UDA_JSON, metric)
    matrices = {0: reference}
    matrices.update({ratio: load_metric_matrix(path, metric) for ratio, path in SSDA_JSONS.items()})
    delta_norm = global_delta_norm()

    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    for ratio, (row, col) in PANEL_ORDER:
        is_reference = ratio == 0
        if ratio != 0 and col == 1:
            show_ylabels = True
        else:
            show_ylabels = ratio == 0
        draw_matrix(
            axes[row, col],
            matrices[ratio],
            reference,
            ratio,
            output["display_name"],
            show_xlabels=(ratio == 0) or (row == 1 and ratio != 0),
            show_ylabels=show_ylabels,
            delta_norm=delta_norm,
        )

    draw_mean_gain_lineplot(axes[1, 0], metric)

    for ax in axes[:, 0]:
        ax.set_anchor("E")
    for ax in axes[:, 1]:
        ax.set_anchor("C")
    for ax in axes[:, 2]:
        ax.set_anchor("W")

    fig.subplots_adjust(left=0.080, right=0.910, top=0.920, bottom=0.150, wspace=0.075, hspace=0.330)

    group_shift = 0.025
    for ax in axes[:, 1:].ravel():
        box = ax.get_position()
        ax.set_position([box.x0 + group_shift, box.y0, box.width, box.height])

    add_bottom_legend(fig)

    fig.canvas.draw()
    left_shift = LEFT_COLUMN_SHIFT_CM / 2.54 / FIG_WIDTH_IN
    baseline_box = axes[0, 0].get_position()
    line_box = axes[1, 0].get_position()
    baseline_x0 = baseline_box.x0 - left_shift
    axes[0, 0].set_position([baseline_x0, baseline_box.y0, baseline_box.width, baseline_box.height])
    axes[1, 0].set_position([baseline_x0, line_box.y0, baseline_box.width, baseline_box.height])

    fig.canvas.draw()
    baseline_pos = axes[0, 0].get_position()
    line_pos = axes[1, 0].get_position()
    top_left = axes[0, 1].get_position()
    top_right = axes[0, 2].get_position()
    bottom_mid = axes[1, 1].get_position()
    bottom_right = axes[1, 2].get_position()
    heatmap_x0 = top_left.x0
    heatmap_x1 = top_right.x1
    heatmap_y0 = min(bottom_mid.y0, bottom_right.y0)
    heatmap_y1 = top_left.y1
    panel_label_shift = PANEL_LABEL_SHIFT_LEFT_CM / 2.54 / FIG_WIDTH_IN

    fig.text(
        baseline_pos.x0 - 0.055,
        (baseline_pos.y0 + baseline_pos.y1) / 2,
        "Source domain",
        ha="center",
        va="center",
        rotation=90,
        fontsize=SHARED_LABEL_FONTSIZE,
    )
    fig.text(
        baseline_pos.x0 - 0.010 - panel_label_shift,
        baseline_pos.y1 + 0.022,
        "(a)",
        ha="right",
        va="center",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="semibold",
    )
    fig.text(
        line_pos.x0 - 0.010 - panel_label_shift,
        line_pos.y1 + 0.022,
        "(b)",
        ha="right",
        va="center",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="semibold",
    )

    label_y = heatmap_y0 - 0.070
    fig.text(
        (baseline_pos.x0 + baseline_pos.x1) / 2,
        baseline_pos.y0 - 0.045,
        "Target domain",
        ha="center",
        va="center",
        fontsize=SHARED_LABEL_FONTSIZE,
    )
    fig.text(
        baseline_pos.x0 - 0.055,
        (line_pos.y0 + line_pos.y1) / 2,
        "Mean Accuracy gain (Δ)" if metric == "accuracy" else "Mean Macro-F1 gain (Δ)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=SHARED_LABEL_FONTSIZE,
    )
    fig.text(
        (line_pos.x0 + line_pos.x1) / 2,
        label_y,
        "Labeled target-domain data (%)",
        ha="center",
        va="center",
        fontsize=SHARED_LABEL_FONTSIZE,
    )

    fig.text(
        heatmap_x0 - 0.050,
        (heatmap_y0 + heatmap_y1) / 2,
        "Source domain",
        ha="center",
        va="center",
        rotation=90,
        fontsize=SHARED_LABEL_FONTSIZE,
    )
    fig.text(
        (heatmap_x0 + heatmap_x1) / 2,
        label_y,
        "Target domain",
        ha="center",
        va="center",
        fontsize=SHARED_LABEL_FONTSIZE,
    )

    title_y = axes[0, 1].get_position().y1 + 0.022
    fig.text(
        heatmap_x0 - 0.020 - (1.0 / 2.54 / FIG_WIDTH_IN),
        title_y,
        "(c)",
        ha="right",
        va="center",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="semibold",
    )

    cbar_ax = fig.add_axes([0.940, heatmap_y0, 0.014, heatmap_y1 - heatmap_y0])
    cbar = fig.colorbar(ScalarMappable(norm=delta_norm, cmap=DELTA_CMAP), cax=cbar_ax)
    cbar_label = "Accuracy gain relative to UDA (Δ)" if metric == "accuracy" else "Macro-F1 gain relative to UDA (Δ)"
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
    cbar.outline.set_linewidth(0.7)

    fig.savefig(output["png"], dpi=300)
    fig.savefig(output["pdf"])
    fig.savefig(output["svg"])
    plt.close(fig)
    print(f"Saved: {output['png']}")
    print(f"Saved: {output['pdf']}")
    print(f"Saved: {output['svg']}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": choose_font_family(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    draw_metric_figure("accuracy")
    draw_metric_figure("macro_f1")


if __name__ == "__main__":
    main()

