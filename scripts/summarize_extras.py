#!/usr/bin/env python3
"""Summarize results/extras.csv: memory, files and single-call latency, ours vs FAISS.

Each value is the median over rounds, with the rounds listed in order.
    .venv/bin/python scripts/summarize_extras.py results/extras.csv
"""
import csv
import statistics
import sys
from collections import defaultdict


def load(path):
    rows, seen = defaultdict(dict), defaultdict(int)
    for r in csv.DictReader(open(path, newline="")):
        key = (r["tag"], r["lib"], r["phase"], r["metric"], r["ef"])
        seen[key] += 1
        label = r["phase"]
        if r["lib"] == "faiss" and r["phase"] == "load" and seen[key] == 2:
            label = "load-mmap"  # the second FAISS load in a round is the IO_FLAG_MMAP run
        try:
            value = float(r["value"])
        except ValueError:
            value = r["value"]
        rows[(r["lib"], label, r["metric"], r["ef"])][r["tag"]] = value
    return rows


def show(rows, lib, label, metric, ef="", fmt="{:,.1f}"):
    values = rows.get((lib, label, metric, ef))
    if not values:
        return "n/a"
    if isinstance(next(iter(values.values())), str):
        return "; ".join(sorted(set(values.values())))
    ordered = [values[t] for t in sorted(values)]
    return f"{fmt.format(statistics.median(ordered))}  ({' / '.join(fmt.format(x) for x in ordered)})"


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "results/extras.csv")
    print("median (rounds A / B / C)\n")
    print("BUILD AND SAVE, 10 build threads")
    for metric, fmt in [("build_s", "{:.1f}"), ("footprint_growth_after_build", "{:,.0f}"),
                        ("rss_growth_after_build", "{:,.0f}"), ("save_s", "{:.2f}"), ("file_mb", "{:,.1f}")]:
        print(f"  {metric:32s} ours {show(rows, 'ours', 'build-save', metric, fmt=fmt):36s} "
              f"faiss {show(rows, 'faiss', 'build-save', metric, fmt=fmt)}")
    for label in ("load", "load-mmap"):
        for lib in ("ours", "faiss"):
            if not any(k[0] == lib and k[1] == label for k in rows):
                continue
            print(f"\nLOAD ({label}), {lib}")
            for metric, fmt in [("load_s", "{:.4f}"), ("footprint_growth_after_load", "{:,.1f}"),
                                ("rss_growth_after_load", "{:,.1f}"), ("first_query_us", "{:,.0f}"),
                                ("first_full_pass_qps", "{:,.0f}"), ("footprint_growth_after_pass", "{:,.1f}"),
                                ("rss_growth_after_pass", "{:,.1f}")]:
                print(f"  {metric:32s} {show(rows, lib, label, metric, fmt=fmt)}")
            if (lib, label, "load_failed", "") in rows:
                print(f"  load_failed                      {show(rows, lib, label, 'load_failed')}")
    for ef in ("60", "140"):
        print(f"\nSINGLE-CALL LATENCY at ef={ef}, microseconds (loaded index, 1 thread, 1 query per call)")
        for lib in ("ours", "faiss"):
            print(f"  {lib:5s} recall {show(rows, lib, 'load', 'recall', ef, '{:.4f}'):34s} "
                  f"p50 {show(rows, lib, 'load', 'latency_p50_us', ef)}")
            print(f"        {'':41s} p99 {show(rows, lib, 'load', 'latency_p99_us', ef)}")
            print(f"        {'':41s} mean {show(rows, lib, 'load', 'latency_mean_us', ef)}")


if __name__ == "__main__":
    main()
