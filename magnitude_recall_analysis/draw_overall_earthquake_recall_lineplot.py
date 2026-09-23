#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw overall earthquake recall versus magnitude from gain_connect.csv.

For each method and magnitude bin, the plotted value is the unweighted mean of
the available transfer-direction recalls.  This reproduces the aggregation
used by the original figure while deriving every value from the input CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

MAGNITUDE_BINS = ("0-1", "1-2", "2-3", ">=3")
METHODS = {
    "full_ft": "FT",
    "dan": "DAN",
    "dann": "DANN",
    "dsan": "DSAN",
}
STYLES = {
    "full_ft": {"color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    "dan": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "dann": {"color": "#D55E00", "marker": "^", "linestyle": "-."},
    "dsan": {"color": "#009E73", "marker": "D", "linestyle": ":"},
}
REQUIRED_COLUMNS = {"task", "magnitude_bin", *METHODS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot transfer-direction-mean earthquake recall by magnitude."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="overall_earthquake_recall_lineplot")
    return parser.parse_args()


def load_and_summarize(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")
    df = df.copy()
    df["magnitude_bin"] = df["magnitude_bin"].astype(str).str.strip()
    invalid_bins = sorted(set(df["magnitude_bin"]) - set(MAGNITUDE_BINS))
    if invalid_bins:
        raise ValueError(f"Invalid magnitude bins: {invalid_bins}")
    if df.duplicated(["task", "magnitude_bin"]).any():
        raise ValueError("Input contains duplicate task/magnitude_bin rows")

    records: list[dict[str, object]] = []
    for magnitude_bin in MAGNITUDE_BINS:
        subset = df[df["magnitude_bin"] == magnitude_bin]
        for method in METHODS:
            values = pd.to_numeric(subset[method], errors="coerce").dropna()
            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{method}/{magnitude_bin} contains values outside [0, 1]")
            records.append(
                {
                    "magnitude_bin": magnitude_bin,
                    "method": method,
                    "method_label": METHODS[method],
                    "mean_recall": float(values.mean()) if len(values) else np.nan,
                    "mean_recall_percent": float(100.0 * values.mean()) if len(values) else np.nan,
                    "n_directions": int(len(values)),
                }
            )
    summary = pd.DataFrame(records)
    if summary["mean_recall"].isna().any():
        missing = summary[summary["mean_recall"].isna()][["magnitude_bin", "method"]]
        raise ValueError(f"No valid recalls for:\n{missing.to_string(index=False)}")
    return summary


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_and_summarize(input_path)

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    x = np.arange(len(MAGNITUDE_BINS))
    for method, label in METHODS.items():
        method_rows = summary[summary["method"] == method].set_index("magnitude_bin")
        values = method_rows.loc[list(MAGNITUDE_BINS), "mean_recall_percent"].to_numpy(float)
        ax.plot(
            x,
            values,
            label=label,
            color=STYLES[method]["color"],
            marker=STYLES[method]["marker"],
            linestyle=STYLES[method]["linestyle"],
            linewidth=1.2 if method == "full_ft" else 1.6,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["0–1", "1–2", "2–3", "≥3"])
    ax.set_xlabel("Magnitude range")
    ax.set_ylabel("Earthquake recall (%)")
    ax.set_ylim(35, 100)
    ax.set_yticks(np.arange(40, 101, 20))
    ax.grid(axis="y", color="#EEEEEE", linestyle="-", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3.5)
    ax.legend(
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.55,
        markerscale=0.82,
    )
    fig.tight_layout(pad=0.8)

    outputs = [output_dir / f"{args.stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / f"{args.stem}_values.csv"
    summary.to_csv(summary_path, index=False)
    for output in [*outputs, summary_path]:
        print(f"[OK] saved: {output}")


if __name__ == "__main__":
    main()
