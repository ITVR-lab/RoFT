#!/usr/bin/env python3
"""Leave-one-dataset-out recipe selection.

For every held-out dataset, a recipe is chosen among candidate tags by the mean accuracy on the
other 17 datasets (never on the held-out dataset itself). Prints, per family and k, the
selection counts and the resulting LOO-selected average, next to each candidate's average.

usage: loo_select.py --family tg=tg,tg_so3,tg_small --family ft=unimts_ft,ft_so3 --k 1 5 --seed 0
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
from collections import Counter

import numpy as np

RES = os.environ.get(
    "ROFT_BENCH_DIR",
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "logs", "bench"),
)


def load(tag, k, s):
    p = os.path.join(RES, "%s_k%d_s%d.json" % (tag, k, s))
    if not os.path.exists(p):
        return None
    return {r["dataset"]: r for r in json.load(open(p))["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="acc")
    a = ap.parse_args()
    out = {}
    for fam in a.family:
        name, tags = fam.split("=")
        tags = tags.split(",")
        for k in a.k:
            runs = {t: load(t, k, a.seed) for t in tags}
            runs = {t: r for t, r in runs.items() if r is not None and len(r) == 18}
            if not runs:
                continue
            ds = sorted(next(iter(runs.values())).keys())
            chosen, sel = Counter(), {m: [] for m in ("acc", "f1", "R@2")}
            for d in ds:
                score = {t: np.mean([r[o][a.metric] for o in ds if o != d]) for t, r in runs.items()}
                best = max(score, key=score.get)
                chosen[best] += 1
                for m in sel:
                    sel[m].append(runs[best][d][m])
            line = "  ".join("%s %.1f/%.1f" % (t, 100 * np.mean([r[d]["acc"] for d in ds]),
                                              100 * np.mean([r[d]["f1"] for d in ds])) for t, r in runs.items())
            print("[%s k=%d] candidates: %s" % (name, k, line))
            print("[%s k=%d] LOO-selected: acc %.1f f1 %.1f R@2 %.1f  choices %s"
                  % (name, k, 100 * np.mean(sel["acc"]), 100 * np.mean(sel["f1"]), 100 * np.mean(sel["R@2"]),
                     dict(chosen)))
            out["%s_k%d" % (name, k)] = {"choices": dict(chosen), **{m: float(np.mean(v)) for m, v in sel.items()}}
    json.dump(out, open(os.path.join(RES, "loo_select_s%d.json" % a.seed), "w"), indent=1)


if __name__ == "__main__":
    main()
