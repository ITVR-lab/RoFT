#!/usr/bin/env python3
"""Wilcoxon signed-rank tests for RoFT vs baselines from JSON bench logs.

Pairs seed-averaged scores per dataset (18 pairs) for each k, then applies
Holm correction across the pooled family of comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def load_rows(path):
    return {r["dataset"]: r for r in json.load(open(path))["rows"]}


def mean_over_seeds(bench_dir, tag, k, seeds):
    by_ds = defaultdict(list)
    for s in seeds:
        p = os.path.join(bench_dir, "%s_k%d_s%d.json" % (tag, k, s))
        if not os.path.isfile(p):
            continue
        for ds, row in load_rows(p).items():
            by_ds[ds].append(row)
    if not by_ds:
        return None
    out = {}
    for ds, rows in by_ds.items():
        out[ds] = {m: float(np.mean([r[m] for r in rows])) for m in ("acc", "f1", "R@2")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", default=os.path.join("logs", "bench"))
    ap.add_argument("--roft", default="ft_small_ens")
    ap.add_argument(
        "--baselines",
        nargs="+",
        default=["unimts_ft", "tip", "probe", "imagebind", "biobank", "gpt4ts"],
    )
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--metric", default="acc", choices=["acc", "f1", "R@2"])
    args = ap.parse_args()

    family_p = []
    family_meta = []
    print("metric=%s  roft=%s" % (args.metric, args.roft))
    for base in args.baselines:
        for k in args.ks:
            a = mean_over_seeds(args.bench_dir, args.roft, k, args.seeds)
            b = mean_over_seeds(args.bench_dir, base, k, args.seeds)
            if a is None or b is None:
                print("[%s k=%d] missing logs — skip" % (base, k))
                continue
            ds = sorted(set(a) & set(b))
            xa = np.array([a[d][args.metric] for d in ds])
            xb = np.array([b[d][args.metric] for d in ds])
            diff = xa - xb
            # zero differences are dropped by scipy wilcoxon by default in recent versions
            try:
                stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = np.nan, 1.0
            wins = int((diff > 0).sum())
            print(
                "[%s k=%d] n=%d  meanΔ=%+.2f  wins=%d/%d  W=%.3g  p=%.3g"
                % (base, k, len(ds), 100 * diff.mean(), wins, len(ds), stat, p)
            )
            family_p.append(p)
            family_meta.append((base, k, p, 100 * diff.mean(), wins, len(ds)))

    if family_p:
        adj = holm(np.array(family_p, dtype=float))
        print("\nHolm-adjusted p-values over %d comparisons:" % len(family_p))
        for (base, k, p, dlt, w, n), pa in zip(family_meta, adj):
            print("  %s k=%d  raw=%.3g  holm=%.3g  Δ=%+.2f  %d/%d" % (base, k, p, pa, dlt, w, n))


if __name__ == "__main__":
    main()
