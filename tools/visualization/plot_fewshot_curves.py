#!/usr/bin/env python3
"""Redraw fig_fewshot_curves.pdf from logs/bench/summary.json (3-seed mean +/- std)."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SUMMARY = os.path.join(ROOT, "logs", "bench", "summary.json")
OUT = os.path.join(ROOT, "outputs", "figures", "fig_fewshot_curves.pdf")

KS = [1, 2, 3, 5, 10]
METRICS = ["acc", "f1", "R@2"]
MNAME = {"acc": "Acc", "f1": "F1", "R@2": "R@2"}
PLOT = [
    ("tip", "Tip-Adapter", "#ba68c8", "-."),
    ("limu", "LIMU-BERT-X", "#00897b", "-."),
    ("units", "UniTS", "#ef6c00", "-."),
    ("onehar", "oneHAR", "#6d4c41", "-."),
    ("gpt4ts", "GPT4TS", "#f5b184", "-."),
    ("biobank", "BioBankSSL", "#c98a5a", "-."),
    ("imagebind", "ImageBind", "#7e57c2", "-."),
    ("unimts_ft", "UniMTS FT", "#1565c0", "-"),
    ("ft_small_ens", "RoFT (Ours)", "#d32f2f", "-"),
]


def main():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    avg = json.load(open(SUMMARY))["avg"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))
    for ax, m in zip(axes, METRICS):
        for t, lab, col, ls in PLOT:
            if t not in avg:
                print("missing", t)
                continue
            mu, sd = [], []
            for k in KS:
                rec = avg[t].get(str(k), {}).get(m)
                if not rec:
                    print("missing", t, k, m)
                    continue
                mu.append(rec[0])
                sd.append(rec[1])
            ks = np.array(KS[: len(mu)], dtype=float)
            mu = np.array(mu)
            sd = np.array(sd)
            ax.fill_between(ks, mu - sd, mu + sd, color=col, alpha=0.28, linewidth=0, zorder=1)
            ax.plot(ks, mu, ls, color=col, marker="o", ms=5.5,
                    lw=2.6 if t == "ft_small_ens" else 1.8, label=lab, zorder=3)
            print("%-14s %s  sd=%s" % (t, m, np.round(sd, 2)))
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([r"$k{=}%d$" % k for k in KS])
        ax.set_xlabel("Number of samples per class")
        ax.set_ylabel(MNAME[m] + " (%)")
        ax.grid(alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(
        h, l, loc="lower center", ncol=len(l), frameon=False,
        handlelength=1.8, handletextpad=0.4, columnspacing=1.05,
        bbox_to_anchor=(0.5, 0.90),
        prop={"family": "Times New Roman", "size": 12},
    )
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.12)
    png = OUT.replace(".pdf", ".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.12, dpi=160)
    plt.close(fig)
    print("wrote", OUT)
    print("wrote", png)


if __name__ == "__main__":
    main()
