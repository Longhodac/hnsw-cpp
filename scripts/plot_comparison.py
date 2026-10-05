#!/usr/bin/env python3
"""Recall versus QPS for our index and FAISS, from results/comparison.csv.

Two panels (1 thread, 10 threads), each with its own y-axis, so no chart has two scales. Each
library keeps one color in both panels. Markers are the median over rounds at each ef, and the
faint band spans the lowest to the highest round. Writes a light and a dark PNG.

    .venv/bin/python scripts/plot_comparison.py results/comparison.csv --out-dir docs

Colors are categorical slots 1 and 2 of the reference palette, checked with the dataviz
validator in both modes (CVD separation 24.7 light, 26.8 dark; contrast above 3:1).
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402

PX = 0.75  # points per CSS pixel
MODES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", ours="#2a78d6", faiss="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", ours="#3987e5", faiss="#d95926"),
}
LABELS = {"ours": "Ours", "faiss": "FAISS"}
MIN_RECALL = 0.88  # ef 10 and 20 fall below this and are cut so the useful range is readable


def load(path):
    cells = defaultdict(lambda: defaultdict(list))  # (lib, threads) -> ef -> [(recall, qps)]
    for row in csv.DictReader(open(path, newline="")):
        cells[(row["lib"], int(row["build_threads"]))][int(row["ef_search"])].append(
            (float(row["recall"]), float(row["qps_median"])))
    series = {}
    for key, by_ef in cells.items():
        points = []
        for ef, runs in sorted(by_ef.items()):
            recall = statistics.median(r for r, _ in runs)
            qps = [q for _, q in runs]
            if recall >= MIN_RECALL:
                points.append((ef, recall, statistics.median(qps), min(qps), max(qps)))
        series[key] = points
    return series


def render(series, mode, out):
    c = MODES[mode]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), facecolor=c["surface"])
    for ax, threads in zip(axes, (1, 10)):
        ax.set_facecolor(c["surface"])
        ax.set_yscale("log")
        for lib in ("ours", "faiss"):
            pts = series[(lib, threads)]
            x = [p[1] for p in pts]
            ax.fill_between(x, [p[3] for p in pts], [p[4] for p in pts], color=c[lib], alpha=0.10, linewidth=0)
            ax.plot(x, [p[2] for p in pts], color=c[lib], linewidth=2 * PX, solid_capstyle="round",
                    solid_joinstyle="round", marker="o", markersize=8 * PX, markerfacecolor=c[lib],
                    markeredgecolor=c["surface"], markeredgewidth=2 * PX, zorder=3)
        # Direct labels at one point of each line, in ink, never in the series color. "Ours" sits
        # above FAISS at every ef, so one label goes above its line and the other below.
        for lib, dy in (("ours", 13), ("faiss", -19)):
            pts = series[(lib, threads)]
            ef, recall, qps = next((p[0], p[1], p[2]) for p in pts if p[0] == 100)
            ax.annotate(LABELS[lib], (recall, qps), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=10, fontweight="medium", color=c["ink2"])
        ax.set_xlim(MIN_RECALL, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.minorticks_off()
        ax.grid(True, which="major", color=c["grid"], linewidth=1 * PX, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["axis"])
        ax.tick_params(colors=c["muted"], labelsize=9, length=0)
        ax.set_xlabel("recall@10", color=c["ink2"], fontsize=10)
        ax.set_ylabel("queries per second, log scale", color=c["ink2"], fontsize=10)
        ax.set_title(f"{threads} thread{'s' if threads > 1 else ''}", loc="left", color=c["ink"],
                     fontsize=12, fontweight="medium", pad=12)
    handles = [Line2D([0], [0], color=c[lib], linewidth=2 * PX, marker="o", markersize=8 * PX,
                      markerfacecolor=c[lib], markeredgecolor=c["surface"], markeredgewidth=2 * PX,
                      label=LABELS[lib]) for lib in ("ours", "faiss")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=10,
               labelcolor=c["ink2"], bbox_to_anchor=(0.985, 1.0))
    fig.text(0.012, 0.012,
             "SIFT1M, M=16, efConstruction=100. Markers are the median of 3 rounds at each ef, the band spans "
             "the lowest to the highest round.\nef 10 and 20 have recall below 0.88 and are cut off. "
             "Each panel has its own y-axis.",
             color=c["muted"], fontsize=8, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=c["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="results/comparison.csv")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args()
    series = load(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        out = out_dir / f"phase7_recall_qps_{mode}.png"
        render(series, mode, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
