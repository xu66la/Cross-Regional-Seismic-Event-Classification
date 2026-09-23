# 随机挑选 earthquake / explosion 各一个样本，同时画出对应的 1D 原始波形和 STFT（2x2）

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ====== 配置 =====
# STFT 原始矩阵根目录（不是 256×256）

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = Path("/path/to/US_EQ_EX")

STFT_ROOT = (
    DATA_ROOT
    / "BASE"
    / "processing_BASE_outputs"
    / "legacy_dataset"
    / "base_dataset_zrt_waveforms_ZRT_DanFenLiang_40HZ_90S_EventFiltered_test_STFT_raw"
)

SAVE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "STFT_visualization"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ("earthquake", "explosion")

SEED = 42
FS = 40
DURATION_SEC = 90
F_MIN, F_MAX = 0.0, FS / 2.0
XTICKS = [0, 30, 60, 90]
STFT_DB_FLOOR = -80.0
WAVE_COLOR = "#1f4e79"
WAVE_LINEWIDTH = 0.9
GRID_COLOR = "#b0b7bf"
GRID_ALPHA = 0.18
GRID_LINESTYLE = "--"
SMOOTH_SIGMA = 0.6

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Arial", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 1.4,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

random.seed(SEED)


def raw_name_from_stft_name(stft_name: str) -> str:
    if not stft_name.endswith("_STFT_raw.npy"):
        raise ValueError(f"文件名不符合预期：{stft_name}")
    return stft_name.replace("_STFT_raw.npy", ".npy")


def waveform_root_from_stft_root(stft_root: str) -> str:
    if not stft_root.endswith("_STFT_raw"):
        raise ValueError(f"STFT 根目录格式不符合预期：{stft_root}")
    return stft_root[:-len("_STFT_raw")]


def load_random_pair(category: str) -> dict:
    stft_dir = os.path.join(STFT_ROOT, category)
    waveform_dir = os.path.join(waveform_root_from_stft_root(STFT_ROOT), category)

    if not os.path.isdir(stft_dir):
        raise RuntimeError(f"STFT 目录不存在：{stft_dir}")
    if not os.path.isdir(waveform_dir):
        raise RuntimeError(f"1D 波形目录不存在：{waveform_dir}")

    stft_files = [f for f in os.listdir(stft_dir) if f.endswith(".npy")]
    if not stft_files:
        raise RuntimeError(f"STFT 目录里没有 .npy 文件：{stft_dir}")

    random.shuffle(stft_files)
    for stft_name in stft_files:
        raw_name = raw_name_from_stft_name(stft_name)
        raw_path = os.path.join(waveform_dir, raw_name)
        stft_path = os.path.join(stft_dir, stft_name)
        if not os.path.isfile(raw_path):
            continue

        waveform = np.load(raw_path)
        stft_mat = np.load(stft_path)

        if waveform.ndim != 1:
            raise ValueError(f"1D 波形不是一维数组：{waveform.shape}")
        if stft_mat.ndim != 2:
            raise ValueError(f"STFT 不是二维矩阵：{stft_mat.shape}")

        print(f"[{category}] STFT 文件：{stft_path}")
        print(f"[{category}] 1D 波形文件：{raw_path}")
        return {
            "category": category,
            "waveform": waveform,
            "stft": stft_mat,
            "raw_name": raw_name,
            "stft_name": stft_name,
        }

    raise FileNotFoundError(f"{category} 目录下没有找到能对应到 1D 波形的 STFT 文件")


def add_panel_label(ax, label: str):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    wf = waveform.astype(np.float32, copy=False)
    std = float(np.std(wf))
    if std < 1e-12:
        return wf - float(np.mean(wf))
    return (wf - float(np.mean(wf))) / std


def stft_to_relative_db(stft_mat: np.ndarray, ref_amp: float) -> np.ndarray:
    eps = 1e-12
    db = 20.0 * np.log10(np.maximum(stft_mat, eps) / max(ref_amp, eps))
    db = np.clip(db, STFT_DB_FLOOR, 0.0)
    return gaussian_filter(db, sigma=SMOOTH_SIGMA)


def style_axes(ax, is_waveform: bool):
    ax.tick_params(length=3, width=0.8, pad=2)
    if is_waveform:
        ax.grid(
            axis="both",
            color=GRID_COLOR,
            alpha=GRID_ALPHA,
            linestyle=GRID_LINESTYLE,
            linewidth=0.6,
        )
    else:
        ax.grid(False)


def main():
    if not os.path.isdir(STFT_ROOT):
        raise RuntimeError(f"STFT 根目录不存在：{STFT_ROOT}")

    samples = [load_random_pair(category) for category in CATEGORIES]
    global_ref = max(float(np.max(sample["stft"])) for sample in samples)
    normalized_waveforms = [normalize_waveform(sample["waveform"]) for sample in samples]
    max_abs = max(float(np.max(np.abs(wf))) for wf in normalized_waveforms)
    waveform_ylim = (-1.05 * max_abs, 1.05 * max_abs)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 7.6),
        dpi=200,
        constrained_layout=True,
        sharex="col",
    )
    panel_labels = [
        "(a) Earthquake waveform",
        "(b) Earthquake STFT",
        "(c) Explosion waveform",
        "(d) Explosion STFT",
    ]
    image_for_cbar = None

    for row, (sample, waveform_norm) in enumerate(zip(samples, normalized_waveforms)):
        t = np.arange(waveform_norm.size) / FS

        ax_wave = axes[row, 0]
        ax_wave.plot(t, waveform_norm, lw=WAVE_LINEWIDTH, color=WAVE_COLOR)
        ax_wave.set_xlabel("Time")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_xlim(0, DURATION_SEC)
        ax_wave.set_ylim(*waveform_ylim)
        ax_wave.set_xticks(XTICKS)
        ax_wave.yaxis.set_major_locator(MaxNLocator(4))
        style_axes(ax_wave, is_waveform=True)
        add_panel_label(ax_wave, panel_labels[row * 2])

        ax_stft = axes[row, 1]
        stft_db = stft_to_relative_db(sample["stft"], global_ref)
        image_for_cbar = ax_stft.imshow(
            stft_db,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[0, DURATION_SEC, F_MIN, F_MAX],
            vmin=STFT_DB_FLOOR,
            vmax=0.0,
            interpolation="bilinear",
        )
        ax_stft.set_xlabel("Time")
        ax_stft.set_ylabel("Frequency")
        ax_stft.set_xticks(XTICKS)
        ax_stft.set_yticks([0, 10, 20])
        style_axes(ax_stft, is_waveform=False)
        add_panel_label(ax_stft, panel_labels[row * 2 + 1])

    cbar = fig.colorbar(
        image_for_cbar,
        ax=axes[:, 1],
        pad=0.02,
        shrink=0.86,
        fraction=0.045,
        aspect=28,
    )
    cbar.set_label("Amplitude (normalized)")

    out_png = os.path.join(SAVE_DIR, "earthquake_explosion_waveform_stft_viz.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    print("✅ 已保存图片：", out_png)


if __name__ == "__main__":
    main()
