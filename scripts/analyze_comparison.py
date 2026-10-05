#!/usr/bin/env python3
"""Summarize results/comparison.csv: build time and QPS at fixed recall, ours vs FAISS.

QPS at a recall target is read off each run's own recall-vs-QPS curve (linear in recall,
log in QPS), so the two libraries are compared at equal recall, not at equal ef. Every round
is reported on its own so the run-to-run spread stays visible.

    .venv/bin/python scripts/analyze_comparison.py results/comparison.csv
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict

TARGETS = (0.90, 0.95, 0.99)
NOISE_MARGIN = 0.15


def qps_at_recall(points, target):
    """points: [(recall, qps)] for one run. Returns interpolated QPS, or None if out of range."""
    pts = sorted(points)
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if r0 <= target <= r1 and r1 > r0:
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return None


def load(path):
    runs = defaultdict(list)  # (lib, threads, tag) -> rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            runs[(row["lib"], int(row["build_threads"]), row["tag"])].append(row)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="results/comparison.csv")
    args = ap.parse_args()
    runs = load(args.csv)
    threads = sorted({t for _, t, _ in runs})
    libs = ("ours", "faiss")

    rounds = sorted({tag for _, _, tag in runs})
    summary = {}  # (lib, threads) -> {"build": {tag: s}, target: {tag: qps}}
    for (lib, t, tag), rows in sorted(runs.items()):
        s = summary.setdefault((lib, t), {"build": {}, **{x: {} for x in TARGETS}})
        s["build"][tag] = float(rows[0]["build_s"])
        pts = [(float(r["recall"]), float(r["qps_median"])) for r in rows]
        for target in TARGETS:
            s[target][tag] = qps_at_recall(pts, target)

    def med(d):
        vals = [v for v in d.values() if v is not None]
        return statistics.median(vals) if vals else float("nan")

    def listing(d, digits=0):
        return " / ".join(f"{d[t]:,.{digits}f}" if d.get(t) is not None else "n/a" for t in rounds)

    def verdict(ratios):
        if all(r >= 1 + NOISE_MARGIN for r in ratios):
            return "ours faster in every round"
        if all(r <= 1 / (1 + NOISE_MARGIN) for r in ratios):
            return "faiss faster in every round"
        return "NOT ESTABLISHED (rounds disagree or are within noise)"

    print(f"rounds: {', '.join(rounds)}   (each ratio below is ours over faiss, or faiss over ours for build)")
    print("\nBUILD TIME, seconds: median of rounds (round by round)")
    for t in threads:
        o, f = summary[("ours", t)]["build"], summary[("faiss", t)]["build"]
        ratios = [f[r] / o[r] for r in rounds if r in o and r in f]
        print(f"  {t:2d} threads  ours {med(o):6.1f} ({listing(o, 1)})   faiss {med(f):6.1f} ({listing(f, 1)})")
        print(f"               faiss/ours per round {' / '.join(f'{x:.2f}x' for x in ratios)}  ->  "
              f"{verdict(ratios).replace('ours faster', 'ours builds faster').replace('faiss faster', 'faiss builds faster')}")

    print("\nQPS AT EQUAL RECALL: median of rounds (round by round)")
    for t in threads:
        for target in TARGETS:
            o, f = summary[("ours", t)][target], summary[("faiss", t)][target]
            ratios = [o[r] / f[r] for r in rounds if o.get(r) and f.get(r)]
            print(f"  {t:2d} threads recall {target:.2f}  ours {med(o):9,.0f} ({listing(o)})   "
                  f"faiss {med(f):9,.0f} ({listing(f)})")
            print(f"               ours/faiss per round {' / '.join(f'{x:.2f}x' for x in ratios)}  "
                  f"(median {statistics.median(ratios):.2f}x)  ->  {verdict(ratios)}")

    print("\nRECALL AT THE SAME EF, ours vs faiss")
    for t in threads:
        gaps = []
        for tag in rounds:
            ours = {int(r["ef_search"]): float(r["recall"]) for r in runs[("ours", t, tag)]}
            fa = {int(r["ef_search"]): float(r["recall"]) for r in runs[("faiss", t, tag)]}
            gaps += [abs(ours[e] - fa[e]) for e in ours if e in fa]
        print(f"  {t:2d} threads  largest recall gap at the same ef, all rounds: {max(gaps):.4f}")


if __name__ == "__main__":
    main()
