#!/usr/bin/env python3
"""Per-class recall bars + gravity spheres on datasets with large RoFT gains.

k=1, three seeds, UniMTS FT / RoFT / uniform SO(3). Classes with fewer than
MIN_TEST test windows are omitted from the bars (they cannot support a
class-wise claim). Writes
  logs/class_sphere/per_class_k1.json
  outputs/figures/fig_class_sphere.pdf
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
from matplotlib import gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools.fewshot_bench as fb
from tools.fewshot_bench import Runner, cached_load, set_seed
from matplotlib import font_manager
from plot_rotation_signal import cone_circle

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_JSON = os.path.join(_ROOT, "logs", "class_sphere", "per_class_k1.json")
OUT_FIG = os.path.join(_ROOT, "outputs", "figures", "fig_class_sphere.pdf")
DATA_PATH = os.environ.get("ROFT_DATA", os.path.join(_ROOT, "UniMTS_data"))
CKPT = os.path.join(_ROOT, "checkpoint", "UniMTS.pth")
SEEDS = (0, 1, 2)
MIN_TEST = 20

# Four largest-gain datasets that contain sitting/standing (k=1, 3-seed table).
DATASETS = ["Harth", "ut-complex", "Shoaib", "MotionSense"]
SHORT = {"Harth": "Harth", "ut-complex": "UT-Complex", "Shoaib": "Shoaib", "MotionSense": "MotionSense"}
JOINT = {"Harth": 9, "ut-complex": 5, "Shoaib": 1, "MotionSense": 5}
POSTURE = ("lying", "sitting", "standing")
POSTURE_COL = {"lying": "#6a1b9a", "sitting": "#ef6c00", "standing": "#2e7d32"}
METHOD_COL = {"unimts_ft": "#1565c0", "ft_small_ens": "#d32f2f", "ft_so3_ens": "#9e9e9e"}
METHOD_LAB = {"unimts_ft": "UniMTS FT", "ft_small_ens": "RoFT", "ft_so3_ens": r"uniform SO(3)"}
METHODS = [
    ("unimts_ft", {"aug": "none", "ens": 0}),
    ("ft_small_ens", {"aug": "small", "ens": 1}),
    ("ft_so3_ens", {"aug": "so3", "ens": 1}),
]


def _score(y_true, pred, logits, y_rank):
    r = _orig_score(y_true, pred, logits, y_rank)
    ncls = int(max(int(y_true.max()), int(pred.max())) + 1)
    rec, nte = [], []
    for c in range(ncls):
        m = y_true == c
        nte.append(int(m.sum()))
        rec.append(float((pred[m] == c).mean()) if m.any() else 0.0)
    r["recall"] = rec
    r["n_test"] = nte
    return r


_orig_score = fb.score
fb.score = _score


def make_args(tag, aug, ens, seed):
    return argparse.Namespace(
        method="unimts_ft", tag=tag, k=1, seed=seed, epochs=200, lr=1e-4,
        lr_cache=1e-3, kl_w=0.1, use_text=1, use_tip=1, use_ce=1,
        batch_size=64, padding_size=200, data_path=DATA_PATH, checkpoint=CKPT,
        out_dir=os.path.join(_ROOT, "logs", "class_sphere"), dump_feats="", aug=aug,
        aug_angle=30.0, head_lr=0.0, ens=ens, stage="finetune", gyro=0, stft=0,
        num_class=2,
    )


def pretty(name):
    n = name.replace("ascending", "up").replace("descending", "down")
    n = n.replace("cycling sitting inactive", "cyc. sit still")
    n = n.replace("cycling standing inactive", "cyc. stand still")
    n = n.replace("cycling sitting", "cyc. sit")
    n = n.replace("cycling standing", "cyc. stand")
    n = n.replace("nordic walking", "nordic walk")
    n = n.replace("vacuum cleaning", "vacuum")
    n = n.replace("rope jumping", "rope jump")
    n = n.replace("stand up", "stand")
    n = n.replace("giving a talk", "talk")
    n = n.replace("drinking coffee", "coffee")
    return n


def posture_key(name):
    n = name.lower().strip()
    if n.startswith("cycling"):
        return None
    if "lying" in n or n == "lay":
        return "lying"
    if "sitting" in n or n == "sit":
        return "sitting"
    if "standing" in n or n in ("stand", "stand up"):
        return "standing"
    return None


def dirs_from_windows(x, joint):
    acc = x[:, :, joint * 3:joint * 3 + 3].numpy()
    g = acc.mean(1)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-8)


def eval_methods(datasets, blob=None):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if blob is None:
        blob = {"k": 1, "seeds": list(SEEDS), "datasets": {}}
    for ds in datasets:
        tr_x, tr_y, names, ncls = cached_load(ds, "train", 1, DATA_PATH, 200)
        te_x, te_y, _, _ = cached_load(ds, "test", None, DATA_PATH, 200)
        names = [str(n) for n in names]
        entry = {
            "names": names, "joint": JOINT[ds], "n_test": None,
            "methods": {t: {"acc": [], "recall": []} for t, _ in METHODS},
        }
        for seed in SEEDS:
            for tag, cfg in METHODS:
                print("====", ds, "seed", seed, tag, cfg, flush=True)
                set_seed(seed)
                args = make_args(tag, cfg["aug"], cfg["ens"], seed)
                runner = Runner(args)
                res = runner.run_official(ds, tr_x, tr_y, te_x, te_y, ncls, "unimts_ft", names)
                used = res.pop("ens") if "ens" in res else res
                entry["n_test"] = used["n_test"]
                entry["methods"][tag]["acc"].append(float(used["acc"]))
                entry["methods"][tag]["recall"].append([float(v) for v in used["recall"]])
                print("   acc=%.4f" % used["acc"], flush=True)
                del runner
                torch.cuda.empty_cache()
        blob["datasets"][ds] = entry
        with open(OUT_JSON, "w") as f:
            json.dump(blob, f, indent=2)
    return blob


def setup_times():
    """Use Times New Roman (or a metric-compatible serif) for every text artist."""
    candidates = [
        "Times New Roman",
        "TimesNewRomanPSMT",
        "TeXGyreTermes",
        "TeX Gyre Termes",
        "Nimbus Roman",
        "Liberation Serif",
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
        "mathtext.rm": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return name


def sphere_wire(ax):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 13)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#bdbdbd", linewidth=0.35, rstride=1, cstride=1)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.42)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
        ax.dist = 8
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=6, pad=-4)
    ax.set_xlabel("$a_x$", fontsize=8, labelpad=-8)
    ax.set_ylabel("$a_y$", fontsize=8, labelpad=-8)
    ax.set_zlabel("$a_z$", fontsize=8, labelpad=-10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#c8c8c8")
        axis.pane.set_alpha(1)


def draw_sphere(ax, te_x, te_y, tr_x, tr_y, names, joint, azim=42):
    sphere_wire(ax)
    y_te, y_tr = te_y.numpy(), tr_y.numpy()
    g_te = dirs_from_windows(te_x, joint)
    g_tr = dirs_from_windows(tr_x, joint)
    rng = np.random.RandomState(0)
    plotted = []
    for c, name in enumerate(names):
        key = posture_key(name)
        if key is None:
            continue
        col = POSTURE_COL[key]
        gt = g_te[y_te == c]
        if len(gt) > 55:
            gt = gt[rng.choice(len(gt), 55, replace=False)]
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=8, c=col, alpha=0.45, depthshade=False)
        gs = g_tr[y_tr == c]
        if len(gs):
            ghat = gs.mean(0)
            ghat = ghat / max(np.linalg.norm(ghat), 1e-8)
            circ = cone_circle(ghat, 30)
            ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color=col, lw=1.4)
            ax.scatter(*ghat, s=42, c=col, marker="*", edgecolors="k", linewidths=0.3, depthshade=False, zorder=5)
            ax.plot([0, ghat[0]], [0, ghat[1]], [0, ghat[2]], color=col, lw=0.8)
            plotted.append(key)
    ax.view_init(elev=18, azim=azim)
    return plotted


def dataset_delta_k1():
    """3-seed mean Acc at k=1 from the official benchmark logs (same as Table 1)."""
    from make_bench_tables import DATASETS as DS, DNAME
    acc = {}
    for t in ("unimts_ft", "ft_small_ens"):
        vals = []
        for s in (0, 1, 2):
            with open(os.path.join(_ROOT, "logs", "bench", "%s_k1_s%d.json" % (t, s))) as f:
                rows = {r["dataset"]: r["acc"] for r in json.load(f)["rows"]}
            vals.append([rows[d] for d in DS])
        acc[t] = np.mean(vals, 0) * 100.0
    return list(DNAME), acc["ft_small_ens"] - acc["unimts_ft"]


def keep_classes(names, n_test, rec_m, max_n=8):
    """Always keep postures; fill remaining slots by |RoFT-FT| among classes with enough tests."""
    n_test = np.asarray(n_test)
    ncls = len(n_test)
    names = list(names[:ncls])
    ok = [i for i in range(ncls) if n_test[i] >= MIN_TEST]
    post = [i for i in ok if posture_key(names[i]) is not None]
    rest = [i for i in ok if i not in post]
    rest.sort(key=lambda i: -abs(rec_m["ft_small_ens"][i] - rec_m["unimts_ft"][i]))
    keep = post + rest[: max(0, max_n - len(post))]
    keep.sort()
    return keep, names


def plot(blob):
    font_name = setup_times()
    cols = [ds for ds in DATASETS if ds in blob["datasets"]]
    nd = len(cols)
    fig = plt.figure(figsize=(13.2, 6.7))
    # One nested column per dataset so the sphere and bars share the same x-span.
    outer = gridspec.GridSpec(
        1, nd, wspace=0.08, left=0.02, right=0.995, top=0.90, bottom=0.14,
    )
    dnames, delta = dataset_delta_k1()
    delta_map = {n: float(d) for n, d in zip(dnames, delta)}
    short_to_delta = {
        "Harth": delta_map.get("Harth", 0),
        "UT-Complex": delta_map.get("UT-Complex", 0),
        "Shoaib": delta_map.get("Shoaib", 0),
        "MotionSense": delta_map.get("MotionSense", 0),
    }
    azim = {"Harth": 28, "ut-complex": 35, "Shoaib": -40, "MotionSense": 50}
    sph_letters = "abcd"
    bar_letters = "efgh"
    sphere_axes, bar_axes = [], []
    for i, ds in enumerate(cols):
        col = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[i], height_ratios=[1.05, 1.22],
            width_ratios=[0.50, 1.0], wspace=0.04, hspace=0.18,
        )
        e = blob["datasets"][ds]
        n_test = np.asarray(e["n_test"])
        rec_m, rec_s, acc_m = {}, {}, {}
        for t in METHOD_LAB:
            rec = np.asarray(e["methods"][t]["recall"]) * 100
            rec_m[t] = rec.mean(0)
            rec_s[t] = rec.std(0)
            acc_m[t] = float(np.mean(e["methods"][t]["acc"]) * 100)
        keep, names = keep_classes(e["names"], n_test, rec_m)

        tr_x, tr_y, _, _ = cached_load(ds, "train", 1, DATA_PATH, 200)
        te_x, te_y, _, _ = cached_load(ds, "test", None, DATA_PATH, 200)
        ax3 = fig.add_subplot(col[0, 1], projection="3d")
        draw_sphere(ax3, te_x, te_y, tr_x, tr_y, names, e["joint"], azim=azim.get(ds, 42))
        dlt = short_to_delta.get(SHORT[ds], acc_m["ft_small_ens"] - acc_m["unimts_ft"])
        ax3.set_title("(%s)  %s  ($%+.1f$ Acc)" % (sph_letters[i], SHORT[ds], dlt),
                      fontsize=10, pad=4, fontname=font_name)
        sphere_axes.append(ax3)

        ax = fig.add_subplot(col[1, 1])
        ypos = np.arange(len(keep))
        h = 0.24
        bar_order = ("unimts_ft", "ft_so3_ens", "ft_small_ens")
        for j, t in enumerate(bar_order):
            ax.barh(
                ypos + (j - 1) * h, rec_m[t][keep], height=h, color=METHOD_COL[t],
                xerr=rec_s[t][keep], error_kw=dict(elinewidth=0.55, capsize=1.2, ecolor="#616161"),
                label=METHOD_LAB[t],
            )
        ax.set_yticks(ypos)
        ax.set_yticklabels([pretty(names[idx]) for idx in keep], fontsize=8, fontname=font_name)
        for tick, idx in zip(ax.get_yticklabels(), keep):
            key = posture_key(names[idx])
            if key is not None:
                tick.set_fontweight("bold")
                tick.set_color(POSTURE_COL[key])
        ax.set_xlim(0, 105)
        ax.tick_params(labelsize=7, pad=2)
        ax.invert_yaxis()
        ax.set_title(
            "(%s)  FT %.0f  /  SO(3) %.0f  /  RoFT %.0f" % (
                bar_letters[i], acc_m["unimts_ft"], acc_m["ft_so3_ens"], acc_m["ft_small_ens"]),
            fontsize=9, pad=3, fontname=font_name,
        )
        ax.set_xlabel("Per-class recall (%)", fontsize=8, fontname=font_name)
        bar_axes.append(ax)

    fig.canvas.draw()
    for ax3, axb in zip(sphere_axes, bar_axes):
        bp, sp = axb.get_position(), ax3.get_position()
        # Same horizontal span as the bar axes so the sphere sits on the column centre.
        cx = bp.x0 + bp.width / 2.0
        w = bp.width
        ax3.set_position([cx - w / 2.0, sp.y0, w, sp.height])

    sph_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POSTURE_COL[k], markersize=7, label=k.capitalize())
        for k in POSTURE
    ]
    sph_handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="#424242",
                              markeredgecolor="k", markersize=8, label=r"$k{=}1$ support / $30^\circ$ cone"))
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COL[t], label=METHOD_LAB[t])
        for t in ("unimts_ft", "ft_so3_ens", "ft_small_ens")
    ]
    fig.legend(sph_handles, [h.get_label() for h in sph_handles], loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.995), prop={"family": font_name, "size": 9})
    fig.legend(bar_handles, [h.get_label() for h in bar_handles], loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.012), prop={"family": font_name, "size": 9})
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    print("wrote", OUT_FIG, "font=", font_name)


def load_blob():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {"k": 1, "seeds": list(SEEDS), "datasets": {}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    blob = load_blob()
    if a.plot_only:
        plot(blob)
        return
    missing = [ds for ds in DATASETS if ds not in blob.get("datasets", {})]
    if a.force:
        blob = eval_methods(DATASETS)
    elif missing:
        blob = eval_methods(missing, blob)
    plot(blob)


if __name__ == "__main__":
    main()
