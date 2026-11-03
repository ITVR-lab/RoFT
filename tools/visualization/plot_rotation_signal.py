#!/usr/bin/env python3
"""Method figure: original / 30-degree / SO(3) accelerometer windows + gravity cone.

Uses the UniMTS UCI-HAR waist channel (total acceleration) and the official k=1
support indices. Writes outputs/figures/fig_rotation_signal.pdf
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import resample

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA = os.environ.get("ROFT_DATA", os.path.join(ROOT, "UniMTS_data"))
OUT = os.path.join(ROOT, "outputs", "figures", "fig_rotation_signal.pdf")
# UCI-HAR: total acceleration (g) -> m/s^2, mapped to joint 9 (waist).
G = 9.80665
STAND, LAY = 4, 5
RNG = np.random.RandomState(0)


def load_waist(split):
    X = np.load(os.path.join(DATA, "UCIHAR", "X_%s.npy" % split)).astype(np.float64)
    y = np.load(os.path.join(DATA, "UCIHAR", "y_%s.npy" % split)).astype(int)
    acc = X[:, :, 6:9] * G  # (N, T, 3) total accel
    t_src = acc.shape[1]
    t_20 = max(1, int(round(t_src * 20.0 / 50.0)))
    acc20 = np.stack([resample(s, t_20) for s in acc], 0)
    return acc20, y, t_20


def support_index(y, k=1):
    idx = torch.load(os.path.join(DATA, "few_shot_data_2", "UCIHAR_k=%d.pth" % k))
    # idx[i] are train indices of unique_labels[i]; labels are 0..5 in order.
    return {int(c): np.asarray(idx[int(c)]) for c in np.unique(y)}


def rotmat(axis, ang):
    u = axis / np.linalg.norm(axis)
    kx = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(ang) * kx + (1.0 - np.cos(ang)) * (kx @ kx)


def apply_R(acc, R):
    return acc @ R.T


def gravity(acc):
    g = acc.mean(0)
    n = np.linalg.norm(g)
    return g, g / max(n, 1e-8)


def perp_axis(g):
    a = np.array([1.0, 0.0, 0.0]) if abs(g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(g, a)
    return u / np.linalg.norm(u)


def cone_circle(ghat, deg=30, n=120):
    a = np.array([1.0, 0.0, 0.0]) if abs(ghat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ghat, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    th = np.deg2rad(deg)
    t = np.linspace(0, 2 * np.pi, n)
    return np.cos(th) * ghat + np.sin(th) * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def sphere_wire(ax):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 13)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#bdbdbd", linewidth=0.35, rstride=1, cstride=1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=6, pad=0)
    ax.set_xlabel("$a_x$", fontsize=8, labelpad=-2)
    ax.set_ylabel("$a_y$", fontsize=8, labelpad=-2)
    ax.set_zlabel("$a_z$", fontsize=8, labelpad=-2)


def scatter_dirs(ax, accs, **kw):
    g = accs.mean(1)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-8)
    if len(g) > 80:
        g = g[RNG.choice(len(g), 80, replace=False)]
    ax.scatter(g[:, 0], g[:, 1], g[:, 2], **kw)


def main():
    acc_te, y_te, t20 = load_waist("test")
    acc_tr, y_tr, _ = load_waist("train")
    sup = support_index(y_tr, k=1)
    # Official k=1 support window of each posture.
    stand = acc_tr[int(sup[STAND][0])]
    lay = acc_tr[int(sup[LAY][0])]

    # One informative draw: rotate about an axis perpendicular to gravity.
    # 30 deg = RoFT; 120 deg is a typical large SO(3) sample that can flip posture.
    gs, gs_h = gravity(stand)
    gl, gl_h = gravity(lay)
    u_s, u_l = perp_axis(gs_h), perp_axis(gl_h)
    R30_s = rotmat(u_s, np.deg2rad(30))
    Rso_s = rotmat(u_s, np.deg2rad(120))
    R30_l = rotmat(u_l, np.deg2rad(30))
    Rso_l = rotmat(u_l, np.deg2rad(120))

    series = {
        "Standing": (stand, apply_R(stand, R30_s), apply_R(stand, Rso_s)),
        "Lying": (lay, apply_R(lay, R30_l), apply_R(lay, Rso_l)),
    }
    cols = ["Original", r"Small-angle ($30^\circ$)", r"Uniform SO(3)"]
    ax_col = ["#d32f2f", "#2e7d32", "#1565c0"]
    ax_lab = [r"$a_x$", r"$a_y$", r"$a_z$"]
    t = np.arange(t20) / 20.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(12.2, 7.6))
    grid = gridspec.GridSpec(
        3, 3, height_ratios=[1.0, 1.0, 1.55], hspace=0.48, wspace=0.26,
        left=0.06, right=0.98, top=0.95, bottom=0.04,
    )

    ymin, ymax = -4, 18
    letters = "abcdef"
    li = 0
    for r, (pose, windows) in enumerate(series.items()):
        for c, w in enumerate(windows):
            ax = fig.add_subplot(grid[r, c])
            for j in range(3):
                ax.plot(t, w[:, j], color=ax_col[j], lw=1.15, label=ax_lab[j] if r == 0 and c == 0 else None)
            ax.axhline(0, color="#e0e0e0", lw=0.6)
            ax.set_xlim(0, t[-1])
            ax.set_ylim(ymin, ymax)
            ax.set_title("%s, %s" % (pose, cols[c]), fontsize=9.5)
            if c == 0:
                ax.set_ylabel(r"Acceleration (m/s$^2$)", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(-0.08, 1.12, r"\textbf{(%s)}" % letters[li] if False else "(%s)" % letters[li],
                    transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
            li += 1
            if r == 0 and c == 0:
                ax.legend(loc="upper right", ncol=3, fontsize=7.5, frameon=False,
                          handlelength=1.2, columnspacing=0.8, borderaxespad=0.1)

    ax3 = fig.add_subplot(grid[2, :], projection="3d")
    sphere_wire(ax3)
    scatter_dirs(ax3, acc_te[y_te == STAND], s=10, c="#81c784", alpha=0.45, depthshade=False, label="Test, standing")
    scatter_dirs(ax3, acc_te[y_te == LAY], s=10, c="#ce93d8", alpha=0.45, depthshade=False, label="Test, lying")
    for ghat, color, lab in ((gs_h, "#2e7d32", "Support standing"), (gl_h, "#6a1b9a", "Support lying")):
        circ = cone_circle(ghat, 30)
        ax3.plot(circ[:, 0], circ[:, 1], circ[:, 2], color=color, lw=1.6, label=r"$30^\circ$ cone" if lab.endswith("standing") else None)
        ax3.scatter(*ghat, s=55, c=color, marker="*", edgecolors="k", linewidths=0.4, depthshade=False, zorder=5)
        ax3.plot([0, ghat[0]], [0, ghat[1]], [0, ghat[2]], color=color, lw=1.0)
    # Highlight the two explicit draws on the standing support.
    g30 = R30_s @ gs_h
    gso = Rso_s @ gs_h
    ax3.scatter(*g30, s=36, c="#d32f2f", marker="o", edgecolors="k", linewidths=0.4, depthshade=False, label=r"Standing $+30^\circ$")
    ax3.scatter(*gso, s=40, c="#424242", marker="X", edgecolors="k", linewidths=0.4, depthshade=False, label=r"Standing + SO(3)")
    # A handful of extra SO(3) samples from standing gravity, to show the sphere is covered.
    so_extra = []
    for _ in range(12):
        q = RNG.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        so_extra.append(R @ gs_h)
    so_extra = np.stack(so_extra)
    ax3.scatter(so_extra[:, 0], so_extra[:, 1], so_extra[:, 2], s=14, c="#9e9e9e", marker="x",
                alpha=0.85, depthshade=False, label=r"SO(3) samples")
    ax3.view_init(elev=16, azim=38)
    ax3.set_title("Mean acceleration direction (gravity in the device frame)", fontsize=9.5, pad=2)
    ax3.text2D(-0.02, 1.02, "(g)", transform=ax3.transAxes, fontsize=10, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#81c784", markersize=7, label="Test standing"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ce93d8", markersize=7, label="Test lying"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2e7d32", markersize=11, markeredgecolor="k", label="Support standing / $30^\\circ$ cone"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#6a1b9a", markersize=11, markeredgecolor="k", label="Support lying / $30^\\circ$ cone"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d32f2f", markeredgecolor="k", markersize=7, label=r"Standing $+30^\circ$"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#424242", markeredgecolor="k", markersize=8, label=r"Standing + SO(3)"),
    ]
    ax3.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.72, 0.98), fontsize=7.5, frameon=True, framealpha=0.92)

    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print("wrote", OUT, "T=%d (%.2fs) stand_g=%s lay_g=%s" % (
        t20, t[-1], np.round(gs, 2), np.round(gl, 2)))


if __name__ == "__main__":
    main()
