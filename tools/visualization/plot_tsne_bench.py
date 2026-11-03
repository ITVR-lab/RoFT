#!/usr/bin/env python3
"""t-SNE of test embeddings: pre-trained (zero-shot) vs. UniMTS FT vs. RoFT (k=5, seed 0)."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FEATS = os.path.join(ROOT, "logs", "bench", "feats")
OUT = os.path.join(ROOT, "outputs", "figures")
DATASETS = [("PAMAP", "PAMAP2"), ("Mhealth", "Mhealth"), ("TNDA-HAR", "TNDA-HAR")]
MODELS = [("frozen", "Pre-trained (zero-shot)"), ("unimts_ft", "UniMTS FT"), ("ft_small", "RoFT")]
K, SEED, MAX_N = 5, 0, 2500


def main():
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(11, 10.5))
    stats = {}
    for i, (ds, dname) in enumerate(DATASETS):
        idx = None
        for j, (tag, mname) in enumerate(MODELS):
            path = os.path.join(FEATS, "%s_%s_k%d_s%d.npz" % (tag, ds, K, SEED))
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if not os.path.exists(path):
                ax.set_title("%s / %s (missing)" % (dname, mname), fontsize=9)
                continue
            d = np.load(path)
            f, y = d["feat"], d["y"]
            if idx is None:
                idx = rng.choice(len(y), min(MAX_N, len(y)), replace=False)
            f, y = f[idx], y[idx]
            sil = float(silhouette_score(f, y, metric="cosine"))
            stats.setdefault(ds, {})[tag] = sil
            emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(f)
            ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab20", s=3, alpha=0.8, linewidths=0)
            ax.set_title("%s / %s\nsilhouette = %.3f" % (dname, mname, sil), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tsne_bench.pdf"), bbox_inches="tight")
    json.dump(stats, open(os.path.join(FEATS, "silhouette.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
