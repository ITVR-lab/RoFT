#!/usr/bin/env python3
"""Bubble chart: Acc vs trainable params, bubble size = adaptation time (fig_eff_acc)."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIG = os.path.join(ROOT, "outputs", "figures", "fig_eff_acc.pdf")
PNG = os.path.join(ROOT, "outputs", "figures", "fig_eff_acc.png")

ROWS = [
    # label, params, time_s, acc, color, zorder
    ("Tip-Adapter", 1.0e4, 0.0, 64.1, "#8e24aa", 4),
    ("LIMU-BERT-X", 4.53e4, 8.1, 35.2, "#00897b", 4),
    ("UniTS", 1.16e6, 24.6, 33.1, "#ef6c00", 4),
    ("GPT4TS", 3.52e6, 8.2, 47.4, "#f9a825", 4),
    ("UniMTS FT", 5.19e6, 162.8, 76.2, "#1565c0", 5),
    ("RoFT (Ours)", 5.19e6, 166.0, 79.6, "#c62828", 6),
    ("oneHAR", 1.005e7, 21.9, 39.1, "#6d4c41", 4),
    ("BioBankSSL", 1.099e7, 10.5, 61.9, "#c98a5a", 4),
    ("ImageBind", 1.961e7, 32.3, 63.0, "#5e35b1", 4),
]

# (dx, dy) in points, ha
OFFSETS = {
    "Tip-Adapter": (8, 6, "left"),
    "LIMU-BERT-X": (8, 8, "left"),
    "UniTS": (9, -7, "left"),
    "GPT4TS": (0, -13, "center"),
    "UniMTS FT": (-10, -12, "right"),
    "RoFT (Ours)": (11, 9, "left"),
    "oneHAR": (8, -10, "left"),
    "BioBankSSL": (0, 11, "center"),
    "ImageBind": (9, -2, "left"),
}


def bubble_s(t):
    return 52.0 + 4.6 * max(float(t), 3.0)


def main():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(3.55, 3.72))
    ax = fig.add_axes([0.145, 0.12, 0.82, 0.655])

    for lab, npar, t, acc, col, z in ROWS:
        ax.scatter(
            [npar], [acc], s=bubble_s(t), c=col, alpha=0.84,
            edgecolors="white", linewidths=0.85, zorder=z,
        )
        dx, dy, ha = OFFSETS[lab]
        ax.annotate(
            lab, (npar, acc), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="center", fontsize=7.4, color="#222222", zorder=7,
            fontweight="bold" if lab.startswith("RoFT") else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlim(6.8e3, 4.2e7)
    ax.set_ylim(27.5, 87)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels([r"$10$K", r"$100$K", r"$1$M", r"$10$M"])
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Acc (%)")
    ax.text(0.03, 0.97, r"$k = 5$", transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#333333")
    ax.grid(True, which="major", alpha=0.28, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Size legend in a strip above the axes, same marker scale.
    axl = fig.add_axes([0.145, 0.80, 0.82, 0.19])
    axl.set_xlim(0, 10)
    axl.set_ylim(-0.55, 1.45)
    axl.axis("off")
    axl.text(0.02, 0.42, "Adaptation time", ha="left", va="center",
             fontsize=7.5, color="#444444")
    xs = [4.15, 6.05, 8.15]
    ts = [10, 30, 160]
    for x, t in zip(xs, ts):
        axl.scatter([x], [0.42], s=bubble_s(t), c="#9e9e9e", alpha=0.88,
                    edgecolors="white", linewidths=0.8, clip_on=False, zorder=3)
        axl.text(x, -0.38, f"{t} s", ha="center", va="bottom", fontsize=7.2, color="#444444")

    fig.savefig(FIG, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(PNG, bbox_inches="tight", pad_inches=0.08, dpi=180)
    plt.close(fig)
    print("wrote", FIG)
    print("wrote", PNG)


if __name__ == "__main__":
    main()
