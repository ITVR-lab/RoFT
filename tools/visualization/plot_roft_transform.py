#!/usr/bin/env python3
"""RoFT core transform, drawn in the UniMTS gravity-pipeline visual language.

Every node is a 3-axis accelerometer window (x red / y green / z blue).
RoFT applies one Rodrigues rotation R_{J_l}, |θ|≤30°, shared over time,
to both the dynamic and gravitational parts of an active joint.

Writes outputs/figures/fig_roft_transform.pdf|.png
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
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA = os.environ.get("ROFT_DATA", os.path.join(ROOT, "UniMTS_data"))
OUT_DIR = os.path.join(ROOT, "outputs", "figures")
OUT_PDF = os.path.join(OUT_DIR, "fig_roft_transform.pdf")
OUT_PNG = os.path.join(OUT_DIR, "fig_roft_transform.png")
OUT_PDF_NT = os.path.join(OUT_DIR, "fig_roft_transform_notitle.pdf")
OUT_PNG_NT = os.path.join(OUT_DIR, "fig_roft_transform_notitle.png")

G = 9.80665
RGB = ("#d62728", "#2ca02c", "#1f77b4")
LW = 1.12
EC = "#111111"


def load_walking():
    x = np.load(os.path.join(DATA, "UCIHAR", "X_train.npy")).astype(np.float64)
    y = np.load(os.path.join(DATA, "UCIHAR", "y_train.npy")).astype(int)
    acc = x[y == 0, :, 6:9] * G
    i = int(np.argmax(acc.std(axis=(1, 2))))
    return acc[i], 50.0


def gravity_lp(acc, fs, cutoff=1.2):
    b, a = butter(3, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, acc, axis=0)


def crop_upsample(sig, fs, t0, dur=0.50, n=140):
    i0 = int(round(t0 * fs))
    i1 = min(len(sig), i0 + int(round(dur * fs)) + 1)
    sl = sig[i0:i1]
    t = np.arange(len(sl)) / fs
    t2 = np.linspace(t[0], t[-1], n)
    out = np.stack([interp1d(t, sl[:, j], kind="cubic")(t2) for j in range(3)], 1)
    return t2, out


def rotmat(axis, ang):
    u = axis / np.linalg.norm(axis)
    kx = np.array(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(ang) * kx + (1.0 - np.cos(ang)) * (kx @ kx)


def perp_axis(g):
    a = np.array([0.0, 0.0, 1.0]) if abs(g[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(g, a)
    return u / np.linalg.norm(u)


def style_ax(ax, ylim, tick_fs=8.5):
    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(labelsize=tick_fs, length=2.6, width=0.7, pad=1.3)
    for s in ax.spines.values():
        s.set_color(EC)
        s.set_linewidth(0.85)
    ax.set_facecolor("white")


def plot_xyz(ax, t, sig):
    for j, c in enumerate(RGB):
        ax.plot(t, sig[:, j], color=c, lw=LW, label=("x", "y", "z")[j],
                solid_capstyle="round")


def cone_circle(ghat, deg=30, n=120):
    a = np.array([1.0, 0.0, 0.0]) if abs(ghat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ghat, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    th = np.deg2rad(deg)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.cos(th) * ghat + np.sin(th) * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def spherical_cap(ghat, deg=30, n_ph=28, n_th=12):
    a = np.array([1.0, 0.0, 0.0]) if abs(ghat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ghat, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    th = np.linspace(0.0, np.deg2rad(deg), n_th)
    ph = np.linspace(0.0, 2.0 * np.pi, n_ph)
    pts = np.zeros((n_th, n_ph, 3))
    for i, t in enumerate(th):
        pts[i] = np.cos(t) * ghat + np.sin(t) * (np.outer(np.cos(ph), e1) + np.outer(np.sin(ph), e2))
    return pts


def slerp_arc(a, b, n=36):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    om = np.arccos(dot)
    if om < 1e-6:
        return np.stack([a, b])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (np.sin((1.0 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def display_pair(g0, g1):
    """Put g0 near the north pole so the 30° cap faces the camera."""
    d0 = np.array([0.08, 0.12, 0.99])
    d0 = d0 / np.linalg.norm(d0)
    n_src = np.cross(g0, g1)
    n_n = np.linalg.norm(n_src)
    if n_n < 1e-8:
        return d0, rotmat(perp_axis(d0), np.deg2rad(30.0)) @ d0
    e1 = np.cross(n_src / n_n, g0)
    e1 = e1 / np.linalg.norm(e1)
    c, s = float(np.dot(g1, g0)), float(np.dot(g1, e1))
    e1d = np.array([0.95, 0.12, -0.12])
    e1d = e1d - np.dot(e1d, d0) * d0
    e1d = e1d / np.linalg.norm(e1d)
    d1 = c * d0 + s * e1d
    return d0, d1 / np.linalg.norm(d1)


def draw_unit_sphere(ax, g0, g1):
    """S^2 glyph: equator, Rodrigues axis, 30° cone, sampled directions, g and Rg."""
    d0, d1 = display_pair(g0, g1)
    u_ax = np.cross(d0, d1)
    u_ax = u_ax / np.linalg.norm(u_ax)

    uu = np.linspace(0.0, 2.0 * np.pi, 24)
    vv = np.linspace(0.0, np.pi, 13)
    ax.plot_wireframe(
        np.outer(np.cos(uu), np.sin(vv)),
        np.outer(np.sin(uu), np.sin(vv)),
        np.outer(np.ones_like(uu), np.cos(vv)),
        color="#c2c2c2", linewidth=0.32, rstride=1, cstride=1, alpha=0.9,
    )
    eq = np.linspace(0.0, 2.0 * np.pi, 90)
    ax.plot(np.cos(eq), np.sin(eq), np.zeros_like(eq), color="#8a8a8a", lw=0.7, alpha=0.85)

    circ = cone_circle(d0, 30, n=48)
    fans = [[[0.0, 0.0, 0.0], circ[i], circ[i + 1]] for i in range(len(circ) - 1)]
    ax.add_collection3d(Poly3DCollection(
        fans, facecolors="#ffcc80", alpha=0.16, edgecolors="none",
    ))
    cap = spherical_cap(d0, 30, n_ph=36, n_th=14)
    ax.plot_surface(
        cap[:, :, 0], cap[:, :, 1], cap[:, :, 2],
        color="#ffb74d", alpha=0.40, linewidth=0, shade=False,
    )
    ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color="#ef6c00", lw=1.65)

    inner = cone_circle(d0, 15, n=40)
    ax.plot(inner[:, 0], inner[:, 1], inner[:, 2], color="#ef6c00", lw=0.6, ls=(0, (1.4, 1.0)), alpha=0.7)

    ax.plot([-u_ax[0], u_ax[0]], [-u_ax[1], u_ax[1]], [-u_ax[2], u_ax[2]],
            color="#6a1b9a", lw=1.15, ls=(0, (2.4, 1.3)))
    ax.scatter(*u_ax, s=14, c="#6a1b9a", edgecolors="k", linewidths=0.2, depthshade=False)
    ax.text(*(u_ax * 1.18), r"$\mathbf{u}$", fontsize=7.6, color="#6a1b9a", ha="left", va="bottom")

    rng = np.random.RandomState(2)
    for p in circ[::6][:-1]:
        if np.dot(p, d1) > 0.97:
            continue
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color="#ef6c00", lw=0.55, alpha=0.45)
        ax.scatter(*p, s=9, c="#ffcc80", edgecolors="#ef6c00", linewidths=0.25, depthshade=False, alpha=0.85)

    for _ in range(10):
        phi = rng.uniform(0.0, 2.0 * np.pi)
        th = rng.uniform(0.0, np.deg2rad(28.0))
        e1 = np.cross(d0, u_ax)
        e1 = e1 / np.linalg.norm(e1)
        q = np.cos(th) * d0 + np.sin(th) * (np.cos(phi) * e1 + np.sin(phi) * u_ax)
        ax.scatter(*q, s=5, c="#ef6c00", alpha=0.35, depthshade=False, linewidths=0)

    def arrow3(vec, color, lw, ms):
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=color, lw=lw)
        ax.scatter(*vec, s=ms, c=color, edgecolors="k", linewidths=0.28, depthshade=False)

    arrow3(d0, "#222222", 1.9, 26)
    arrow3(d1, "#d62728", 1.9, 26)
    arc = slerp_arc(d0, d1, n=40)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="#ef6c00", lw=1.35)
    ax.scatter(*arc[-1], s=12, c="#ef6c00", edgecolors="k", linewidths=0.2, depthshade=False)
    mid = arc[int(0.58 * len(arc))]
    ax.text(mid[0] * 1.22 + 0.10, mid[1] * 1.05, mid[2] * 1.08, r"$30^\circ$",
            fontsize=7.6, color="#e65100", ha="left", va="bottom")
    ax.text(-0.42, -0.55, 0.22,
            r"$\mathbf{g}$", fontsize=7.8, color="#222222", ha="center", va="center")
    ax.text(d1[0] * 1.16 + 0.08, d1[1] * 1.08, d1[2] * 0.92,
            r"$\mathbf{Rg}$", fontsize=7.8, color="#d62728", ha="left", va="center")

    ax.view_init(elev=18, azim=-52)
    ax.set_proj_type("ortho")
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.62)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
        ax.dist = 5.4
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_zlim(-1.02, 1.02)
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))


def render(with_titles=True):
    # Target print size from the layout sheet: 15.26 cm × 7.79 cm.
    W_CM, H_CM = 15.26, 7.79
    W_IN, H_IN = W_CM / 2.54, H_CM / 2.54

    acc_full, fs = load_walking()
    grav_full = gravity_lp(acc_full, fs)
    dyn_full = acc_full - grav_full
    ghat = grav_full.mean(0)
    R = rotmat(perp_axis(ghat), np.deg2rad(30.0))
    dyn_r_full = dyn_full @ R.T
    grav_r_full = grav_full @ R.T
    out_full = dyn_r_full + grav_r_full

    energy = (np.diff(acc_full, axis=0) ** 2).sum(1)
    n = int(round(0.50 * fs))
    t0 = np.argmax(np.convolve(energy, np.ones(n), mode="valid")) / fs
    t, acc = crop_upsample(acc_full, fs, t0)
    _, dyn = crop_upsample(dyn_full, fs, t0)
    _, grav = crop_upsample(grav_full, fs, t0)
    _, dyn_r = crop_upsample(dyn_r_full, fs, t0)
    _, grav_r = crop_upsample(grav_r_full, fs, t0)
    _, out = crop_upsample(out_full, fs, t0)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.85,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(W_IN, H_IN))

    gap = 0.042
    left, right = 0.025, 0.990
    usable = right - left - 4 * gap
    w_tall, w_stack, w_sph = 0.168, 0.178, 0.255
    scale = usable / (w_tall + w_stack + w_sph + w_stack + w_tall)
    w_tall *= scale
    w_stack *= scale
    w_sph *= scale

    x0 = left
    x1 = x0 + w_tall + gap
    x2 = x1 + w_stack + gap
    x3 = x2 + w_sph + gap
    x4 = x3 + w_stack + gap

    # Without titles, reclaim the top title band for the axes.
    y_bot, y_top = (0.085, 0.880) if with_titles else (0.085, 0.955)
    h_tall = y_top - y_bot
    gap_v = 0.100
    h_stack = 0.5 * (h_tall - gap_v)
    y_mid = y_bot + h_stack + gap_v

    raw = (x0, y_bot, w_tall, h_tall)
    ddyn = (x1, y_mid, w_stack, h_stack)
    dgrv = (x1, y_bot, w_stack, h_stack)
    sph = (x2, y_bot + 0.02, w_sph, h_tall - 0.04)
    rdyn = (x3, y_mid, w_stack, h_stack)
    rgrv = (x3, y_bot, w_stack, h_stack)
    add = (x4, y_bot, w_tall, h_tall)

    pad = 1.8
    y_raw = (min(acc.min(), out.min()) - pad, max(acc.max(), out.max()) + pad)
    y_dyn = (min(dyn.min(), dyn_r.min()) - pad, max(dyn.max(), dyn_r.max()) + pad)
    y_g = (min(grav.min(), grav_r.min()) - pad, max(grav.max(), grav_r.max()) + pad)

    title_fs, stack_fs, tick_fs, leg_fs = 8.0, 7.2, 6.8, 7.0

    ax_raw = fig.add_axes(raw)
    plot_xyz(ax_raw, t, acc)
    style_ax(ax_raw, y_raw, tick_fs=tick_fs)
    if with_titles:
        ax_raw.set_title("1. raw input", fontsize=title_fs, pad=3.0, fontweight="bold")
    ax_raw.legend(
        loc="lower left", fontsize=leg_fs, frameon=False,
        handlelength=1.15, handletextpad=0.30, borderaxespad=0.20, labelspacing=0.12,
    )

    ax_dd = fig.add_axes(ddyn)
    plot_xyz(ax_dd, t, dyn)
    style_ax(ax_dd, y_dyn, tick_fs=tick_fs)
    ax_dd.tick_params(labelbottom=False)
    if with_titles:
        ax_dd.set_title("2. zero-gravity acceleration", fontsize=stack_fs, pad=2.2, fontweight="bold")

    ax_dg = fig.add_axes(dgrv)
    plot_xyz(ax_dg, t, grav)
    style_ax(ax_dg, y_g, tick_fs=tick_fs)
    if with_titles:
        ax_dg.set_title("3. gravitational acceleration", fontsize=stack_fs, pad=2.2, fontweight="bold")

    ax_rd = fig.add_axes(rdyn)
    plot_xyz(ax_rd, t, dyn_r)
    style_ax(ax_rd, y_dyn, tick_fs=tick_fs)
    ax_rd.tick_params(labelbottom=False)
    if with_titles:
        ax_rd.set_title("4. dynamic rotation", fontsize=stack_fs, pad=2.2, fontweight="bold")

    ax_rg = fig.add_axes(rgrv)
    plot_xyz(ax_rg, t, grav_r)
    style_ax(ax_rg, y_g, tick_fs=tick_fs)
    if with_titles:
        ax_rg.set_title("5. gravity rotation", fontsize=stack_fs, pad=2.2, fontweight="bold")

    g0 = grav.mean(0)
    g0 = g0 / np.linalg.norm(g0)
    g1 = grav_r.mean(0)
    g1 = g1 / np.linalg.norm(g1)
    ax_sp = fig.add_axes(sph, projection="3d")
    draw_unit_sphere(ax_sp, g0, g1)
    ax_sp.set_title(
        r"same $\mathbf{R}_{J_l}$  on  $\mathbb{S}^2$",
        fontsize=8.0, pad=1.5, color=EC, fontweight="bold",
    )

    ax_out = fig.add_axes(add)
    plot_xyz(ax_out, t, out)
    style_ax(ax_out, y_raw, tick_fs=tick_fs)
    if with_titles:
        ax_out.set_title("6. augment output", fontsize=title_fs, pad=3.0, fontweight="bold")

    out_pdf = OUT_PDF if with_titles else OUT_PDF_NT
    out_png = OUT_PNG if with_titles else OUT_PNG_NT
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("wrote", out_pdf)


def main():
    render(with_titles=True)
    render(with_titles=False)


if __name__ == "__main__":
    main()
