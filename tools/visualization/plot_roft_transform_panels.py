#!/usr/bin/env python3
"""Export the seven RoFT-transform panels at the exact composite sizes.

Wave panels: same geometry / colors / tick labels as the whole figure, but no
titles or legends. Unit-sphere panel keeps its labels.
Writes outputs/figures/roft_transform_panels/.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tools.visualization.plot_roft_transform as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_DIR = os.path.join(ROOT, "outputs", "figures", "roft_transform_panels")

# Same canvas as plot_roft_transform.main()
W_CM, H_CM = 15.26, 7.79
W_IN, H_IN = W_CM / 2.54, H_CM / 2.54
DPI = 300


def layout_sizes():
    """Reproduce fractional panel sizes from the composite, return inches."""
    gap = 0.042
    left, right = 0.025, 0.990
    usable = right - left - 4 * gap
    w_tall, w_stack, w_sph = 0.168, 0.178, 0.255
    scale = usable / (w_tall + w_stack + w_sph + w_stack + w_tall)
    w_tall *= scale
    w_stack *= scale
    w_sph *= scale

    y_bot, y_top = 0.085, 0.820
    h_tall = y_top - y_bot
    gap_v = 0.095
    h_stack = 0.5 * (h_tall - gap_v)
    h_sph = h_tall - 0.04

    return {
        "tall": (w_tall * W_IN, h_tall * H_IN),
        "stack": (w_stack * W_IN, h_stack * H_IN),
        "sphere": (w_sph * W_IN, h_sph * H_IN),
    }


def setup_rc():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.85,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "stix",
    })


def save_wave(path_stem, t, sig, ylim, figsize):
    fig = plt.figure(figsize=figsize)
    # Leave room for tick labels; no title/legend.
    ax = fig.add_axes([0.22, 0.16, 0.74, 0.78])
    P.plot_xyz(ax, t, sig)
    P.style_ax(ax, ylim, tick_fs=7.2)
    fig.savefig(path_stem + ".pdf", dpi=DPI)
    fig.savefig(path_stem + ".png", dpi=DPI)
    plt.close(fig)
    print("wrote", path_stem + ".pdf", "in", tuple(round(x, 4) for x in figsize))


def save_sphere(path_stem, g0, g1, figsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88], projection="3d")
    P.draw_unit_sphere(ax, g0, g1)
    ax.set_title(
        r"same $\mathbf{R}_{J_l}$  on  $\mathbb{S}^2$",
        fontsize=9.2, pad=1.5, color=P.EC, fontweight="bold",
    )
    fig.savefig(path_stem + ".pdf", dpi=DPI)
    fig.savefig(path_stem + ".png", dpi=DPI)
    plt.close(fig)
    print("wrote", path_stem + ".pdf", "in", tuple(round(x, 4) for x in figsize))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_rc()
    sizes = layout_sizes()

    acc_full, fs = P.load_walking()
    grav_full = P.gravity_lp(acc_full, fs)
    dyn_full = acc_full - grav_full
    ghat = grav_full.mean(0)
    R = P.rotmat(P.perp_axis(ghat), np.deg2rad(30.0))
    dyn_r_full = dyn_full @ R.T
    grav_r_full = grav_full @ R.T
    out_full = dyn_r_full + grav_r_full

    energy = (np.diff(acc_full, axis=0) ** 2).sum(1)
    n = int(round(0.50 * fs))
    t0 = np.argmax(np.convolve(energy, np.ones(n), mode="valid")) / fs
    t, acc = P.crop_upsample(acc_full, fs, t0)
    _, dyn = P.crop_upsample(dyn_full, fs, t0)
    _, grav = P.crop_upsample(grav_full, fs, t0)
    _, dyn_r = P.crop_upsample(dyn_r_full, fs, t0)
    _, grav_r = P.crop_upsample(grav_r_full, fs, t0)
    _, out = P.crop_upsample(out_full, fs, t0)

    pad = 1.8
    y_raw = (min(acc.min(), out.min()) - pad, max(acc.max(), out.max()) + pad)
    y_dyn = (min(dyn.min(), dyn_r.min()) - pad, max(dyn.max(), dyn_r.max()) + pad)
    y_g = (min(grav.min(), grav_r.min()) - pad, max(grav.max(), grav_r.max()) + pad)

    g0 = grav.mean(0)
    g0 = g0 / np.linalg.norm(g0)
    g1 = grav_r.mean(0)
    g1 = g1 / np.linalg.norm(g1)

    save_wave(os.path.join(OUT_DIR, "01_raw_acceleration"), t, acc, y_raw, sizes["tall"])
    save_wave(os.path.join(OUT_DIR, "02_zero_gravity_acceleration"), t, dyn, y_dyn, sizes["stack"])
    save_wave(os.path.join(OUT_DIR, "03_gravitational_acceleration"), t, grav, y_g, sizes["stack"])
    save_sphere(os.path.join(OUT_DIR, "04_unit_sphere"), g0, g1, sizes["sphere"])
    save_wave(os.path.join(OUT_DIR, "05_dynamic_rotation"), t, dyn_r, y_dyn, sizes["stack"])
    save_wave(os.path.join(OUT_DIR, "06_gravity_rotation"), t, grav_r, y_g, sizes["stack"])
    save_wave(os.path.join(OUT_DIR, "07_gravity_addition"), t, out, y_raw, sizes["tall"])

    print("out_dir", OUT_DIR)
    for k, (w, h) in sizes.items():
        print(k, "cm", round(w * 2.54, 3), "x", round(h * 2.54, 3))


if __name__ == "__main__":
    main()
