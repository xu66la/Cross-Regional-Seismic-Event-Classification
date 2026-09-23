"""Draw a CONUS overview map with approximate research-region boxes."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUT_DIR = PROJECT_ROOT / "outputs" / "region_visualization"
OUT_SVG = OUT_DIR / "us_research_regions_cartopy.svg"


MAP_EXTENT = (-128, -65, 24, 50)

# Approximate boxes for visual reference only: (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "MSH":  {"bbox": (-124.0, -120.0, 45.0, 48.0), "color": "#D55E00", "label": (-125.4, 49.2), "arrow": (-122.0, 48.0), "linewidth": 2.0, "linestyle": "-"},
    "IDOR": {"bbox": (-121.0, -111.0, 43.0, 47.0), "color": "#0072B2", "label": (-118.5, 49.0), "arrow": (-116.0, 47.0), "linewidth": 1.2, "linestyle": "--"},
    "HLP":  {"bbox": (-123.0, -114.0, 40.0, 46.0), "color": "#009E73", "label": (-109.7, 40.8), "arrow": (-114.0, 42.0), "linewidth": 2.0, "linestyle": "-"},
    "SPE":  {"bbox": (-120.0, -114.0, 34.0, 40.0), "color": "#CC79A7", "label": (-109.8, 36.5), "arrow": (-114.0, 36.7), "linewidth": 1.2, "linestyle": "--"},
    "SSIP": {"bbox": (-119.0, -114.0, 30.0, 36.0), "color": "#E69F00", "label": (-111.1, 30.9), "arrow": (-114.0, 33.0), "linewidth": 1.2, "linestyle": "--"},
    "BASE": {"bbox": (-112.0, -102.0, 42.0, 48.0), "color": "#56B4E9", "label": (-99.7, 47.1), "arrow": (-102.0, 45.0), "linewidth": 2.0, "linestyle": "-"},
    "ENAM": {"bbox": (-84.0, -74.0, 32.0, 40.0), "color": "#332288", "label": (-70.5, 39.0), "arrow": (-74.0, 36.0), "linewidth": 2.0, "linestyle": "-"},
    "GASC": {"bbox": (-86.0, -80.0, 29.0, 36.0), "color": "#117733", "label": (-75.8, 29.5), "arrow": (-80.0, 31.4), "linewidth": 1.2, "linestyle": "--"},
}


def draw_region(ax, name, spec):
    lon0, lon1, lat0, lat1 = spec["bbox"]
    color = spec["color"]

    ax.add_patch(
        Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor=color,
            linewidth=spec["linewidth"],
            linestyle=spec["linestyle"],
            zorder=5,
        )
    )

    tx, ty = spec["arrow"]
    lx, ly = spec["label"]
    ax.annotate(
        name,
        xy=(tx, ty),
        xytext=(lx, ly),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="black",
        arrowprops={
            "arrowstyle": spec.get("arrowstyle", "->"),
            "color": "black",
            "linewidth": 1.2,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=6,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "savefig.dpi": 600,
        }
    )

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = plt.axes(projection=proj)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # A restrained terrain-like background suitable for print.
    ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d8e7ef", alpha=0.58, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2efe6", alpha=0.34, zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#d8e7ef", edgecolor="none", alpha=0.75, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.65, edgecolor="#4d4d4d", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.55, edgecolor="#6b6b6b", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35, edgecolor="#8c8c8c", alpha=0.85, zorder=3)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="#8a8a8a",
        alpha=0.45,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#333333"}
    gl.ylabel_style = {"size": 8, "color": "#333333"}

    for name, spec in REGIONS.items():
        draw_region(ax, name, spec)

    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.075, top=0.985)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_SVG}")


if __name__ == "__main__":
    main()
