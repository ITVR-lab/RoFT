#!/usr/bin/env python3
"""Unsmoothed support-set CE curves at k=10 on UCI-HAR (seed 0).

The original benchmark jobs stored only final metrics, so this is a short
diagnostic re-run of the eight trainable main-table methods on one dataset.
Writes
  logs/loss_curves/ucihar_k10_s0.json
  outputs/figures/fig_loss_curves.pdf
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.fewshot_bench import (
    EXT_METHODS,
    Runner,
    acc_from_graph,
    augment,
    cached_load,
    set_seed,
    to_graph,
)

from matplotlib.colors import to_hex, to_rgb

# User palette (navy → sky → peach → crimson).
SWATCH = [
    "#073068", "#206FB6", "#6BADD7", "#C5DAEE",
    "#FDDFD0", "#FC9171", "#EE3B2A", "#A60E16",
]
ACADEMIC_COL = {
    "limu": SWATCH[0],
    "units": SWATCH[1],
    "onehar": SWATCH[2],
    "gpt4ts": SWATCH[3],
    "biobank": SWATCH[4],
    "imagebind": SWATCH[5],
    "unimts_ft": SWATCH[6],
    "ft_small_ens": SWATCH[7],
}


def stroke_of(hex_c):
    """Darken very light swatches so the line stays visible on white."""
    r, g, b = to_rgb(hex_c)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum < 0.68:
        return hex_c
    t = 0.50
    return to_hex((r * t, g * t, b * t))
ACADEMIC_LAB = {
    "limu": "LIMU-BERT-X",
    "units": "UniTS",
    "onehar": "oneHAR",
    "gpt4ts": "GPT4TS",
    "biobank": "BioBankSSL",
    "imagebind": "ImageBind",
    "unimts_ft": "UniMTS FT",
    "ft_small_ens": "RoFT",
}

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_JSON = os.path.join(_ROOT, "logs", "loss_curves", "ucihar_k10_s0.json")
OUT_FIG = os.path.join(_ROOT, "outputs", "figures", "fig_loss_curves.pdf")
DATA_PATH = os.environ.get("ROFT_DATA", os.path.join(_ROOT, "UniMTS_data"))
CKPT = os.path.join(_ROOT, "checkpoint", "UniMTS.pth")
DS, K, SEED, EPOCHS = "UCIHAR", 10, 0, 200

# Main-table methods that actually train (Tip-Adapter is training-free).
RUNS = [
    ("limu", "limu", "none"),
    ("units", "units", "none"),
    ("onehar", "onehar", "none"),
    ("gpt4ts", "gpt4ts", "none"),
    ("biobank", "biobank", "none"),
    ("imagebind", "imagebind", "none"),
    ("unimts_ft", "unimts_ft", "none"),
    ("ft_small_ens", "unimts_ft", "small"),
]


def setup_times():
    candidates = [
        "Times New Roman", "TimesNewRomanPSMT", "TeXGyreTermes",
        "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    name = next((c for c in candidates if c in available), None)
    if name is None:
        for p in (
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
        ):
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                break
    if name is None:
        name = "serif"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [name, "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return name


def make_args(tag, method, aug):
    return argparse.Namespace(
        method=method, tag=tag, k=K, seed=SEED, epochs=EPOCHS, lr=1e-4,
        lr_cache=1e-3, kl_w=0.1, use_text=1, use_tip=1, use_ce=1,
        batch_size=64, padding_size=200, data_path=DATA_PATH, checkpoint=CKPT,
        out_dir=os.path.join(_ROOT, "logs", "loss_curves"), dump_feats="", aug=aug,
        aug_angle=30.0, head_lr=0.0, ens=0, stage="finetune", gyro=0, stft=0,
        num_class=2,
    )


def train_curve(tag, method, aug, tr_x, tr_y, ncls):
    set_seed(SEED)
    args = make_args(tag, method, aug)
    loader = DataLoader(
        TensorDataset(tr_x, tr_y),
        batch_size=min(args.batch_size, max(1, len(tr_x))),
        shuffle=True,
    )
    if method in EXT_METHODS:
        from tools.baselines.ext_backbones import build
        m = build(method, ncls).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)

        def logits(xb):
            xb = acc_from_graph(augment(to_graph(xb), args.aug, args.aug_angle))
            return m(xb).float()
    else:
        runner = Runner(args)
        m = runner.reset(ncls)
        for n, p in m.named_parameters():
            p.requires_grad = ("model.acc" in n) or n.startswith("fc.")
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=args.lr)

        def logits(xb):
            return m.classifier(augment(to_graph(xb), args.aug, args.aug_angle)).float()

    assert torch.cuda.is_available(), "CUDA required"
    print("device", torch.cuda.get_device_name(0), "mem", torch.cuda.get_device_properties(0).total_memory, flush=True)
    hist = []
    for ep in range(EPOCHS):
        m.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            yb = yb.cuda()
            loss = F.cross_entropy(logits(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(yb)
            n += len(yb)
        hist.append(tot / max(n, 1))
        if (ep + 1) % 20 == 0 or ep == 0:
            print("  ep %d/%d CE=%.4f" % (ep + 1, EPOCHS, hist[-1]), flush=True)
    del m
    torch.cuda.empty_cache()
    return hist


def collect():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    blob = {"dataset": DS, "k": K, "seed": SEED, "epochs": EPOCHS, "curves": {}}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            blob = json.load(f)
    tr_x, tr_y, names, ncls = cached_load(DS, "train", K, DATA_PATH, 200)
    print("support", DS, "k=%d" % K, "N=%d" % len(tr_y), "C=%d" % ncls, flush=True)
    for tag, method, aug in RUNS:
        if tag in blob.get("curves", {}) and len(blob["curves"][tag]) == EPOCHS:
            print("skip", tag, flush=True)
            continue
        print("====", tag, method, aug, flush=True)
        try:
            hist = train_curve(tag, method, aug, tr_x, tr_y, ncls)
        except Exception:
            import traceback
            traceback.print_exc()
            with open(OUT_JSON, "w") as f:
                json.dump(blob, f, indent=1)
            continue
        blob.setdefault("curves", {})[tag] = [float(v) for v in hist]
        with open(OUT_JSON, "w") as f:
            json.dump(blob, f, indent=1)
        print("   last CE=%.4f" % hist[-1], flush=True)
    return blob


def moving_average(y, w=7):
    y = np.asarray(y, dtype=float)
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, np.ones(w) / w, mode="valid")


def plot(blob):
    font_name = setup_times()
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 4.55), sharex=True)
    xs = np.arange(1, EPOCHS + 1)
    letters = "abcdefgh"
    for i, (tag, method, aug) in enumerate(RUNS):
        ax = axes[i // 4, i % 4]
        lab, fill = ACADEMIC_LAB[tag], ACADEMIC_COL[tag]
        line = stroke_of(fill)
        y = np.asarray(blob["curves"][tag], dtype=float)
        ys = moving_average(y, 7)
        fill_a = 0.38 if fill.upper() in ("#C5DAEE", "#FDDFD0") else 0.13
        ax.fill_between(xs, y, color=fill, alpha=fill_a, linewidth=0)
        ax.plot(xs, y, color=line, lw=0.7, alpha=0.32, solid_capstyle="round")
        ax.plot(xs, ys, color=line, ls="-", lw=1.85, alpha=0.96, solid_capstyle="round")
        ymax = max(np.percentile(y, 98), ys.max()) * 1.08
        ax.set_ylim(0, ymax)
        ax.set_xlim(1, EPOCHS)
        ax.set_xticks([1, 100, 200])
        ax.tick_params(labelsize=7, length=3)
        ax.grid(True, axis="y", linewidth=0.35, color="#ececec")
        ax.text(0.04, 0.93, "(%s)  %s" % (letters[i], lab), transform=ax.transAxes,
                fontsize=9, fontname=font_name, fontweight="bold", va="top", color="#212121")
        ax.text(0.96, 0.93, "final %.2f" % y[-1], transform=ax.transAxes,
                fontsize=7.5, fontname=font_name, ha="right", va="top", color="#616161")
        if i // 4 == 1:
            ax.set_xlabel("Epoch", fontsize=9, fontname=font_name)
        if i % 4 == 0:
            ax.set_ylabel("Support CE", fontsize=9, fontname=font_name)
    fig.tight_layout(w_pad=0.55, h_pad=0.55)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    print("wrote", OUT_FIG, "font=", font_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.plot_only:
        with open(OUT_JSON) as f:
            blob = json.load(f)
    else:
        blob = collect()
    plot(blob)


if __name__ == "__main__":
    main()
