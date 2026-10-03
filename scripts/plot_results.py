#!/usr/bin/env python3
"""Plot recall@k (x) vs QPS (y, log scale) from the eval CSV.

    scripts/plot_results.py results/results.csv -o results/recall_qps.png
    scripts/plot_results.py results/results.csv --dataset sift
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="results/results.csv")
    ap.add_argument("-o", "--output", default="results/recall_qps.png")
    ap.add_argument("--dataset", help="only plot this dataset")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    series = defaultdict(list)  # (dataset, label) -> [(recall, qps)]
    k = None
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if args.dataset and row["dataset"] != args.dataset:
                continue
            k = row["k"]
            label = row["index"]
            if row["index"] == "hnsw":
                label += f" M={row['M']} efC={row['ef_construction']}"
            series[(row["dataset"], label)].append((float(row["recall"]), float(row["qps"])))
    if not series:
        raise SystemExit(f"no rows to plot in {args.csv}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for (dataset, label), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"{dataset}: {label}")
    ax.set_yscale("log")
    ax.set_xlabel(f"recall@{k}")
    ax.set_ylabel("queries / second (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
