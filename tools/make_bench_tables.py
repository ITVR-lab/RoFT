#!/usr/bin/env python3
"""Build UniMTS-style LaTeX tables and figures from fewshot_bench.py outputs.

Reads  logs/bench/{tag}_k{k}_s{seed}.json
Writes outputs/tables/*.tex, outputs/figures/*.pdf, logs/bench/summary.json
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

def _project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    for p in (root, here):
        if os.path.isdir(os.path.join(p, "logs", "bench")):
            return p
    return root


ROOT = _project_root()
RES = os.path.join(ROOT, "logs", "bench")
TAB = os.path.join(ROOT, "outputs", "tables")
FIG = os.path.join(ROOT, "outputs", "figures")
KS = [1, 2, 3, 5, 10]
METRICS = ["acc", "f1", "R@2"]
MNAME = {"acc": "Acc", "f1": "F1", "R@2": "R@2"}

DATASETS = [
    "Opp_g", "UCIHAR", "MotionSense", "w-HAR", "Shoaib", "har70plus",
    "realworld", "TNDA-HAR", "PAMAP", "USCHAD", "Mhealth", "Harth",
    "ut-complex", "Wharf", "WISDM", "DSADS", "UTD-MHAD", "MMAct",
]
DNAME = ["Opportunity", "UCI-HAR", "MotionSense", "w-HAR", "Shoaib", "HAR70+",
         "RealWorld", "TNDA-HAR", "PAMAP2", "USC-HAD", "Mhealth", "Harth",
         "UT-Complex", "Wharf", "WISDM", "DSADS", "UTD-MHAD", "MMAct"]
NCLS = [4, 6, 6, 7, 7, 7, 8, 8, 12, 12, 12, 12, 13, 14, 18, 19, 27, 35]

MAIN = [
    ("tip", "Tip-Adapter~\\cite{tipadapter}"),
    ("limu", "LIMU-BERT-X~\\cite{limubertx}"),
    ("units", "UniTS~\\cite{units}"),
    ("onehar", "oneHAR~\\cite{onehar}"),
    ("gpt4ts", "GPT4TS~\\cite{gpt4ts}"),
    ("biobank", "BioBankSSL~\\cite{biobankssl}"),
    ("imagebind", "ImageBind~\\cite{imagebind}"),
    ("unimts_ft", "UniMTS FT~\\cite{unimts}"),
    ("ft_small_ens", "\\textbf{RoFT}"),
]
# Development / diagnostic runs: kept in the appendix, not the main tables.
EXTRA = [
    ("zeroshot", "Zero-shot~\\cite{unimts}"),
    ("nn1", "1-NN"),
    ("proto", "ProtoNet~\\cite{protonet}"),
    ("probe", "Linear Probe~\\cite{unimts}"),
    ("tip_f", "Tip-Adapter-F~\\cite{tipadapter}"),
    ("random", "Random init."),
    ("tg_small_ens", "TipGuided"),
]
ALL = MAIN + EXTRA
# seed-0 ablations of RoFT
ABL = [
    ("ft_small_ens", "RoFT"),
    ("ft_small", "w/o prior ensemble"),
    ("ft_noaug_ens", "w/o rotation perturbation"),
    ("ft_so3_ens", "SO(3) instead of small-angle rotation"),
    ("tg_small_ens", "+ text \\& Tip-cache priors (TipGuided)"),
    ("ft_noaug", "w/o both (UniMTS FT)"),
]
ANGLES = [("ft_noaug_ens", 0), ("ft_a15_ens", 15), ("ft_small_ens", 30), ("ft_a45_ens", 45),
          ("ft_a60_ens", 60), ("ft_so3_ens", 180)]
OURS = "ft_small_ens"

# Transcribed from UniMTS (NeurIPS 2024) Table 1 (zero-shot) and Table 2 (full-shot).
REPORTED = {
    "zs": {
        "acc": [45.9, 35.2, 45.2, 59.0, 63.6, 68.2, 43.6, 59.1, 47.2, 30.5, 70.7, 68.9, 34.8, 18.2, 27.8, 31.5, 22.8, 10.2],
        "f1": [42.2, 22.0, 33.7, 42.9, 57.2, 34.8, 36.7, 53.7, 43.6, 27.8, 61.8, 41.1, 29.2, 13.7, 25.5, 23.7, 18.5, 10.0],
        "R@2": [80.0, 53.1, 57.2, 60.7, 82.1, 86.6, 64.0, 77.5, 63.2, 45.4, 78.7, 85.0, 44.2, 38.6, 47.1, 46.0, 32.6, 18.7],
    },
    "full": {
        "acc": [88.2, 92.1, 99.8, 98.4, 99.7, 96.9, 84.0, 99.6, 98.0, 58.3, 97.0, 98.0, 99.1, 82.7, 81.3, 94.3, 85.6, 77.1],
        "f1": [89.1, 92.2, 99.8, 98.8, 99.7, 87.9, 81.2, 99.6, 98.1, 63.1, 97.3, 86.7, 99.2, 51.7, 80.9, 94.2, 83.2, 71.7],
    },
}
REPORTED_AVG = {("zs", "acc"): (43.5, 17.9), ("zs", "f1"): (34.3, 14.2), ("zs", "R@2"): (58.9, 19.3),
                ("full", "acc"): (90.6, 10.6), ("full", "f1"): (87.5, 13.4)}
REF_ROWS = [("zs", "UniMTS zero-shot$^\\dagger$"), ("full", "UniMTS full-shot$^\\dagger$")]
FIG4_JSON = os.path.join(ROOT, "logs", "unimts_fig4.json")
# Recovered UniMTS Figure 4 curves (keys match extract_unimts_fig4.py / DNAME).
FIG4_ROWS = [
    ("GPT4TS", "GPT4TS~\\cite{gpt4ts}"),
    ("IMUGPT", "IMUGPT~\\cite{imugpt}"),
    ("IMU2CLIP", "IMU2CLIP~\\cite{imu2clip}"),
    ("BioBankSSL", "BioBankSSL~\\cite{biobankssl}"),
    ("Random", "Random init."),
    ("ImageBind", "ImageBind~\\cite{imagebind}"),
    ("UniMTS", "UniMTS FT~\\cite{unimts}"),
]
# Table/figure: dagger only for methods we cannot re-run, plus recovered UniMTS FT as a reference.
FIG4_DAGGER_ROWS = [
    ("IMUGPT", "IMUGPT~\\cite{imugpt}"),
    ("IMU2CLIP", "IMU2CLIP~\\cite{imu2clip}"),
    ("UniMTS", "UniMTS FT~\\cite{unimts}"),
]
FIG4_RERUN_ROWS = [
    ("gpt4ts", "GPT4TS~\\cite{gpt4ts}"),
    ("biobank", "BioBankSSL~\\cite{biobankssl}"),
    ("imagebind", "ImageBind~\\cite{imagebind}"),
    ("unimts_ft", "UniMTS FT (re-run)"),
    ("ft_small_ens", "\\textbf{RoFT}"),
]
FIG4_PLOT = {
    "IMUGPT": ("IMUGPT$^\\dagger$", "#7eb8dc", "--"),
    "IMU2CLIP": ("IMU2CLIP$^\\dagger$", "#f0b84e", "--"),
    "UniMTS": ("UniMTS FT$^\\dagger$", "#e57373", ":"),
}
PLOT = {
    "tip": ("Tip-Adapter", "#ba68c8", "-."),
    "limu": ("LIMU-BERT-X", "#00897b", "-."),
    "units": ("UniTS", "#ef6c00", "-."),
    "onehar": ("oneHAR", "#6d4c41", "-."),
    "gpt4ts": ("GPT4TS", "#f5b184", "-."),
    "biobank": ("BioBankSSL", "#c98a5a", "-."),
    "imagebind": ("ImageBind", "#7e57c2", "-."),
    "unimts_ft": ("UniMTS FT", "#1565c0", "-"),
    "ft_small_ens": ("RoFT (Ours)", "#d32f2f", "-"),
}


def load_all():
    """data[tag][k][seed] = {dataset: row}"""
    data = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(RES, "*_k*_s*.json")):
        m = re.match(r"(.+)_k(\d+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        tag, k, s = m.group(1), int(m.group(2)), int(m.group(3))
        d = json.load(open(path))
        rows = {r["dataset"]: r for r in d["rows"]}
        if all(ds in rows for ds in DATASETS):
            data[tag][k][s] = rows
    return data


SEEDS = None  # restrict to these seeds when set (ablation tables use seed 0)


def mat(data, tag, k, metric):
    """(n_seeds, 18) array in percent, or None."""
    runs = data.get(tag, {}).get(k, {})
    if SEEDS is not None:
        runs = {s: r for s, r in runs.items() if s in SEEDS}
    if not runs:
        return None
    return np.array([[100.0 * runs[s][ds][metric] for ds in DATASETS] for s in sorted(runs)])


def rank_marks(values, higher=True):
    """Return dict idx -> 'best'/'second' using values rounded to 1 decimal."""
    v = [None if x is None else round(x, 1) for x in values]
    uniq = sorted({x for x in v if x is not None}, reverse=higher)
    out = {}
    for i, x in enumerate(v):
        if x is None:
            continue
        if uniq and x == uniq[0]:
            out[i] = "best"
        elif len(uniq) > 1 and x == uniq[1]:
            out[i] = "second"
    return out


def fmt(s, mark):
    if mark == "best":
        return "\\textbf{%s}" % s
    if mark == "second":
        return "\\underline{%s}" % s
    return s


HEADER = r"""\resizebox{\textwidth}{!}{%%
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\renewcommand{\arraystretch}{0.95}
\begin{tabular}{ll|cccccccc|cccccccc|cc|c}
\toprule
Dataset & Metrics & %s & \rotatebox{90}{\small Average} \\
\midrule
Number of Classes & & %s & \\
Level & & \multicolumn{8}{c|}{Easy} & \multicolumn{8}{c|}{Medium} & \multicolumn{2}{c|}{Hard} & Avg \\
\midrule
"""


def wide_table(data, k, methods, refs=True):
    present = [(t, n) for t, n in methods if mat(data, t, k, "acc") is not None]
    if not present:
        return None
    head = HEADER % (" & ".join("\\rotatebox{90}{\\small %s}" % d for d in DNAME),
                     " & ".join(str(c) for c in NCLS))
    cells = {}
    for t, _ in present:
        for m in METRICS:
            a = mat(data, t, k, m).mean(0)
            cells[(t, m)] = (list(a), float(a.mean()), float(a.std()))
    lines = []
    marks = {}
    for m in METRICS:
        for j in range(len(DATASETS) + 1):
            vals = [(cells[(t, m)][0][j] if j < len(DATASETS) else cells[(t, m)][1]) for t, _ in present]
            for i, mk in rank_marks(vals).items():
                marks[(present[i][0], m, j)] = mk
    for pi, (t, name) in enumerate(present):
        nseed = len(data[t][k])
        for mi, m in enumerate(METRICS):
            per, avg, sd = cells[(t, m)]
            row = ["\\multirow{3}{*}{%s}" % name if mi == 0 else "", MNAME[m]]
            row += [fmt("%.1f" % v, marks.get((t, m, j))) for j, v in enumerate(per)]
            row.append(fmt("%.1f(%.1f)" % (avg, sd), marks.get((t, m, len(DATASETS)))))
            lines.append(" & ".join(row) + " \\\\")
        if pi < len(present) - 1:
            lines.append("\\midrule")
    for key, name in REF_ROWS if refs else []:
        lines.append("\\midrule")
        ms = [m for m in METRICS if m in REPORTED[key]]
        for mi, m in enumerate(ms):
            row = ["\\multirow{%d}{*}{\\color{gray}%s}" % (len(ms), name) if mi == 0 else "", "\\color{gray}" + MNAME[m]]
            row += ["\\color{gray}%.1f" % v for v in REPORTED[key][m]]
            row.append("\\color{gray}%.1f(%.1f)" % REPORTED_AVG[(key, m)])
            lines.append(" & ".join(row) + " \\\\")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n}\n"


def seed_avg(data, tag, k, metric):
    """mean and std over seeds of the 18-dataset average."""
    a = mat(data, tag, k, metric)
    if a is None:
        return None
    per_seed = a.mean(1)
    return float(per_seed.mean()), float(per_seed.std()), len(per_seed)


def avg_table(data, methods, metrics=("acc", "f1", "R@2")):
    present = [(t, n) for t, n in methods if any(mat(data, t, k, "acc") is not None for k in KS)]
    cols = [(k, m) for k in KS for m in metrics]
    vals = {(t, c): seed_avg(data, t, c[0], c[1]) for t, _ in present for c in cols}
    marks = {}
    for c in cols:
        v = [vals[(t, c)][0] if vals[(t, c)] else None for t, _ in present]
        for i, mk in rank_marks(v).items():
            marks[(present[i][0], c)] = mk
    nm = len(metrics)
    out = ["\\resizebox{\\textwidth}{!}{%", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{l|" + "|".join(["c" * nm] * len(KS)) + "}", "\\toprule",
           "Shots per class & " + " & ".join("\\multicolumn{%d}{c%s}{$k=%d$}" % (nm, "|" if i < len(KS) - 1 else "", k)
                                             for i, k in enumerate(KS)) + " \\\\",
           "Metrics & " + " & ".join(MNAME[m] for _ in KS for m in metrics) + " \\\\", "\\midrule"]
    for t, name in present:
        row = [name]
        for c in cols:
            v = vals[(t, c)]
            if v is None:
                row.append("--")
                continue
            s = "%.1f" % v[0] if v[2] == 1 else "%.1f$_{\\pm %.1f}$" % (v[0], v[1])
            row.append(fmt(s, marks.get((t, c))))
        out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "}"]
    return "\n".join(out) + "\n"


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(pvals) - r) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def wilcoxon_table(data, summary):
    base = [(t, n) for t, n in ALL if t != OURS]
    rows = []
    for metric in ("acc", "f1"):
        raw, items = [], []
        for t, name in base:
            diffs, per_k = [], {}
            for k in KS:
                a, b = mat(data, OURS, k, metric), mat(data, t, k, metric)
                if a is None or b is None:
                    continue
                d = a.mean(0) - b.mean(0)
                diffs.append(d)
                per_k[k] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            if not diffs:
                continue
            d = np.concatenate(diffs)
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            items.append((t, name, per_k, float(d.mean()), int((d > 0).sum()), len(d)))
            raw.append(p)
        if not raw:
            continue
        adj = holm(np.array(raw))
        for (t, name, per_k, md, wins, n), p, pa in zip(items, raw, adj):
            rows.append((metric, name, per_k, md, wins, n, p, pa))
            summary.setdefault("wilcoxon", {}).setdefault(metric, {})[t] = {
                "pooled_p": p, "holm_p": float(pa), "mean_diff": md, "wins": wins, "n": n, "per_k": per_k}
    out = ["\\begin{tabular}{ll|ccccc|ccc}", "\\toprule",
           "Metric & Baseline & " + " & ".join("$k=%d$" % k for k in KS) + " & $\\Delta$ & W/T & Holm $p$ \\\\",
           "\\midrule"]
    last = None
    for metric, name, per_k, md, wins, n, p, pa in rows:
        if last is not None and metric != last:
            out.append("\\midrule")
        last = metric
        def pf(x):
            return "%.1e" % x if x < 0.01 else "%.2f" % x
        out.append(" & ".join([MNAME[metric], name] + [pf(per_k[k]) if k in per_k else "--" for k in KS]
                              + ["%+.1f" % md, "%d/%d" % (wins, n), pf(pa)]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def efficiency_table(data, k=5):
    rows = []
    for t, name in MAIN:
        runs = data.get(t, {}).get(k, {})
        if not runs:
            continue
        r0 = [runs[s][ds] for s in runs for ds in DATASETS]
        tr = np.mean([r.get("train_s", 0.0) for r in r0])
        npar = np.mean([r.get("n_params", 0) for r in r0])
        rows.append("%s & %s & %.1f \\\\" % (name, "0" if npar == 0 else "%.2fM" % (npar / 1e6)
                                            if npar >= 1e5 else "%.1fK" % (npar / 1e3), tr))
    return ("\\begin{tabular}{lcc}\n\\toprule\nMethod & Trainable params & Adaptation time (s / dataset) \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def angle_table(data):
    out = ["\\begin{tabular}{l|cc|cc}", "\\toprule",
           "Max.\\ rotation angle & \\multicolumn{2}{c|}{$k=1$} & \\multicolumn{2}{c}{$k=5$} \\\\",
           " & Acc & F1 & Acc & F1 \\\\", "\\midrule"]
    for tag, ang in ANGLES:
        row = ["$0^\\circ$ (none)" if ang == 0 else ("SO(3)" if ang == 180 else "$%d^\\circ$" % ang)]
        for k in (1, 5):
            for m in ("acc", "f1"):
                v = seed_avg(data, tag, k, m)
                row.append("--" if v is None else ("\\textbf{%.1f}" if tag == OURS else "%.1f") % v[0])
        out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


DEV = [  # (base tag, description); "_ens" adds the prior ensemble
    ("ft_noaug", "UniMTS FT (official recipe)"),
    ("ft_so3", "+ uniform SO(3) rotation"),
    ("ft_small", "+ small-angle rotation ($30^\\circ$)"),
    ("ft_e30", "30 epochs"),
    ("tg", "TipGuided"),
    ("tg_so3", "TipGuided + SO(3) rotation"),
    ("tg_small", "TipGuided + small-angle rotation"),
    ("tg_hlr", "TipGuided, head lr $10^{-3}$"),
    ("tg_e30", "TipGuided, 30 epochs"),
]


def dev_table(data, ks=(1, 5)):
    """Seed-0 development grid with leave-one-dataset-out selection counts."""
    cands = [t + s for t, _ in DEV for s in ("", "_ens")]
    cands = [t for t in cands if all(mat(data, t, k, "acc") is not None for k in ks)]
    counts = {}
    for k in ks:
        accs = {t: mat(data, t, k, "acc").mean(0) for t in cands}
        for j in range(len(DATASETS)):
            best = max(cands, key=lambda t: np.delete(accs[t], j).mean())
            counts[(best, k)] = counts.get((best, k), 0) + 1
    out = ["\\begin{tabular}{l|" + "|".join(["cc"] * len(ks)) + "|" + "c" * len(ks) + "}", "\\toprule",
           "Candidate & " + " & ".join("\\multicolumn{2}{c|}{$k=%d$ (Acc)}" % k for k in ks)
           + " & \\multicolumn{%d}{c}{LOO selections}" % len(ks) + " \\\\",
           " & " + " & ".join("w/o ens. & w/ ens." for _ in ks) + " & " + " & ".join("$k=%d$" % k for k in ks)
           + " \\\\", "\\midrule"]
    for t, name in DEV:
        if t + "_ens" not in cands and t not in cands:
            continue
        row = [name]
        for k in ks:
            for v in (t, t + "_ens"):
                row.append("%.1f" % seed_avg(data, v, k, "acc")[0] if v in cands else "--")
        row += [str(counts.get((t, k), 0) + counts.get((t + "_ens", k), 0)) for k in ks]
        out.append(" & ".join(row) + " \\\\")
        if t == "ft_e30":
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n", counts


LEVELS = [("Easy", slice(0, 8)), ("Medium", slice(8, 16)), ("Hard", slice(16, 18))]
LEVEL_METHODS = ["tip", "imagebind", "unimts_ft", "ft_small_ens"]


def level_table(data, ks=(1, 5, 10), metric="acc"):
    """Average accuracy per difficulty level; also returns macros \\Rl{tag}{k}{level}."""
    names = dict(ALL)
    present = [t for t in LEVEL_METHODS if all(mat(data, t, k, metric) is not None for k in ks)]
    cols = [(k, lv, sl) for k in ks for lv, sl in LEVELS]
    vals = {(t, k, lv): float(mat(data, t, k, metric).mean(0)[sl].mean()) for t in present for k, lv, sl in cols}
    marks = {}
    for k, lv, _ in cols:
        for i, mk in rank_marks([vals[(t, k, lv)] for t in present]).items():
            marks[(present[i], k, lv)] = mk
    out = ["\\begin{tabular}{l|" + "|".join(["ccc"] * len(ks)) + "}", "\\toprule",
           "Shots per class & " + " & ".join("\\multicolumn{3}{c%s}{$k=%d$}" % ("|" if i < len(ks) - 1 else "", k)
                                             for i, k in enumerate(ks)) + " \\\\",
           "Level & " + " & ".join(lv for _ in ks for lv, _ in LEVELS) + " \\\\", "\\midrule"]
    for t in present:
        out.append(" & ".join([names[t].replace("\\textbf{RoFT}", "RoFT")] +
                              [fmt("%.1f" % vals[(t, k, lv)], marks.get((t, k, lv))) for k, lv, _ in cols]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    macros = ["\\providecommand{\\Rl}[3]{\\csname l@#1@#2@#3\\endcsname}"]
    macros += ["\\expandafter\\def\\csname l@%s@%d@%s\\endcsname{%.1f}" % (t, k, lv, v) for (t, k, lv), v in vals.items()]
    if OURS in present and "unimts_ft" in present:
        macros += ["\\expandafter\\def\\csname l@gain@%d@%s\\endcsname{%+.1f}" % (k, lv, vals[(OURS, k, lv)] - vals[("unimts_ft", k, lv)])
                   for k, lv, _ in cols]
    return "\n".join(out) + "\n", "\n".join(macros) + "\n"


def load_fig4():
    if not os.path.exists(FIG4_JSON):
        return None
    return json.load(open(FIG4_JSON))


def fig4_mat(fig4, method):
    """(18,) array of macro-F1 for each k, stacked as (5, 18)."""
    return np.array([[fig4[method][d]["mean"][i] for d in DNAME] for i in range(len(KS))])


def reported_fewshot_table(data, fig4):
    """Average macro-F1 vs k: leftover UniMTS Figure 4 (dagger) plus our re-runs."""
    rows = []
    vals = {}
    for m, name in FIG4_DAGGER_ROWS:
        mu = fig4_mat(fig4, m).mean(1)
        vals[m] = list(mu)
        rows.append((m, name + "$^\\dagger$", list(mu), True))
    n_dagger = len(rows)
    for t, name in FIG4_RERUN_ROWS:
        if not seed_avg(data, t, 1, "f1"):
            continue
        mu = [seed_avg(data, t, k, "f1")[0] for k in KS]
        vals[t] = mu
        rows.append((t, name, mu, False))
    # Keep recovered macros for every Figure-4 method (appendix recovered-vs-rerun checks).
    for m, _ in FIG4_ROWS:
        if m not in vals:
            vals[m] = list(fig4_mat(fig4, m).mean(1))
    marks = {}
    for j in range(len(KS)):
        for i, mk in rank_marks([r[2][j] for r in rows]).items():
            marks[(rows[i][0], j)] = mk
    out = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l|" + "c" * len(KS) + "}", "\\toprule",
           "Method & " + " & ".join("$k=%d$" % k for k in KS) + " \\\\", "\\midrule"]
    for i, (t, name, mu, dag) in enumerate(rows):
        if i == n_dagger:
            out.append("\\midrule")
        cells = [fmt("%.1f" % v, marks.get((t, j))) for j, v in enumerate(mu)]
        out.append(" & ".join([name] + cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "}"]
    macros = ["\\providecommand{\\Rf}[2]{\\csname f4@#1@#2\\endcsname}"]
    for t, mu in vals.items():
        for k, v in zip(KS, mu):
            macros.append("\\expandafter\\def\\csname f4@%s@%d\\endcsname{%.1f}" % (t, k, v))
    if OURS in vals and "UniMTS" in vals:
        for k, a, b in zip(KS, vals[OURS], vals["UniMTS"]):
            macros.append("\\expandafter\\def\\csname f4@gain@%d\\endcsname{%+.1f}" % (k, a - b))
    return "\n".join(out) + "\n", "\n".join(macros) + "\n"


def plot_sota_f1(data, fig4):
    """Average macro-F1: leftover recovered Figure-4 methods vs our re-runs."""
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for m, (lab, col, ls) in FIG4_PLOT.items():
        mu = fig4_mat(fig4, m).mean(1)
        ax.plot(KS, mu, ls, color=col, marker="s", ms=3.5, lw=1.3, label=lab)
    rerun = [
        ("gpt4ts", PLOT["gpt4ts"]),
        ("biobank", PLOT["biobank"]),
        ("imagebind", PLOT["imagebind"]),
        ("unimts_ft", ("UniMTS FT (re-run)", "#1565c0", "-")),
        (OURS, ("RoFT (ours)", "#d32f2f", "-")),
    ]
    for t, (lab, col, ls) in rerun:
        pts = [(k, seed_avg(data, t, k, "f1")) for k in KS]
        pts = [(k, v) for k, v in pts if v]
        if not pts:
            continue
        ks = np.array([p[0] for p in pts])
        mu = np.array([p[1][0] for p in pts])
        sd = np.array([p[1][1] for p in pts])
        ax.plot(ks, mu, ls, color=col, marker="o", ms=4.5, lw=2.2 if t == OURS else 1.8, label=lab)
        ax.fill_between(ks, mu - sd, mu + sd, color=col, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Number of samples per class")
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_fewshot_sota.pdf"), bbox_inches="tight")
    plt.close(fig)


def write_numbers(data, summary, prefix="r", fname="numbers.tex"):
    """LaTeX macros: \\R{tag}{k}{metric} -> mean over seeds, \\Rsd{...} -> std, \\Rp{tag}{metric} -> Holm p,
    \\Rd{a}{b}{k}{metric} -> difference a-b. With prefix 'z' (seed 0 only): \\Z{tag}{k}{metric}, \\Zd{...}."""
    if prefix == "z":
        out = ["% auto-generated by make_bench_tables.py (seed 0 only)",
               "\\providecommand{\\Z}[3]{\\csname z@#1@#2@#3\\endcsname}",
               "\\providecommand{\\Zd}[4]{\\csname zd@#1@#2@#3@#4\\endcsname}"]
        for t in data:
            for k in KS:
                for m in METRICS:
                    v = seed_avg(data, t, k, m)
                    if v:
                        out.append("\\expandafter\\def\\csname z@%s@%d@%s\\endcsname{%.1f}" % (t, k, m, v[0]))
                for m in METRICS:
                    vals = [seed_avg(data, t, kk, m) for kk in KS]
                    if all(vals):
                        out.append("\\expandafter\\def\\csname z@%s@avg@%s\\endcsname{%.1f}"
                                   % (t, m, np.mean([v[0] for v in vals])))
        for t, _ in ABL + [(a, None) for a, _ in ANGLES]:
            for b, _ in ABL + [(a, None) for a, _ in ANGLES]:
                for k in KS:
                    for m in METRICS:
                        va, vb = seed_avg(data, t, k, m), seed_avg(data, b, k, m)
                        if va and vb:
                            out.append("\\expandafter\\def\\csname zd@%s@%s@%d@%s\\endcsname{%+.1f}"
                                       % (t, b, k, m, va[0] - vb[0]))
        open(os.path.join(TAB, fname), "w").write("\n".join(out) + "\n")
        return
    out = ["% auto-generated by make_bench_tables.py",
           "\\providecommand{\\R}[3]{\\csname r@#1@#2@#3\\endcsname}",
           "\\providecommand{\\Rsd}[3]{\\csname s@#1@#2@#3\\endcsname}",
           "\\providecommand{\\Rd}[4]{\\csname d@#1@#2@#3@#4\\endcsname}",
           "\\providecommand{\\Rp}[2]{\\csname p@#1@#2\\endcsname}",
           "\\providecommand{\\Rw}[2]{\\csname w@#1@#2\\endcsname}"]
    tags = list(data.keys())
    for t in tags:
        for k in KS:
            for m in METRICS:
                v = seed_avg(data, t, k, m)
                if v:
                    out.append("\\expandafter\\def\\csname r@%s@%d@%s\\endcsname{%.1f}" % (t, k, m, v[0]))
                    out.append("\\expandafter\\def\\csname s@%s@%d@%s\\endcsname{%.1f}" % (t, k, m, v[1]))
            for m in METRICS:
                vals = [seed_avg(data, t, k, m) for k in KS]
                if all(vals):
                    out.append("\\expandafter\\def\\csname r@%s@avg@%s\\endcsname{%.1f}"
                               % (t, m, np.mean([v[0] for v in vals])))
    for t in [x for x in (OURS, "ft_small", "tg_small_ens", "unimts_ft") if x in tags]:
        for b in tags:
            for m in METRICS:
                for k in KS:
                    va, vb = seed_avg(data, t, k, m), seed_avg(data, b, k, m)
                    if va and vb:
                        out.append("\\expandafter\\def\\csname d@%s@%s@%d@%s\\endcsname{%+.1f}"
                                   % (t, b, k, m, va[0] - vb[0]))
    for m, d in summary.get("wilcoxon", {}).items():
        for t, w in d.items():
            p = w["holm_p"]
            s = "%.3f" % p if p >= 1e-3 else "%.1f\\times10^{%d}" % (p / 10 ** np.floor(np.log10(p)), np.floor(np.log10(p)))
            out.append("\\expandafter\\def\\csname p@%s@%s\\endcsname{%s}" % (t, m, s))
            out.append("\\expandafter\\def\\csname w@%s@%s\\endcsname{%d/%d}" % (t, m, w["wins"], w["n"]))
    open(os.path.join(TAB, "numbers.tex"), "w").write("\n".join(out) + "\n")


def _times_rc():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_curves(data):
    _times_rc()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))
    for ax, m in zip(axes, METRICS):
        for t, (lab, col, ls) in PLOT.items():
            pts = [(k, seed_avg(data, t, k, m)) for k in KS]
            pts = [(k, v) for k, v in pts if v]
            if not pts:
                continue
            ks = np.array([p[0] for p in pts], dtype=float)
            mu = np.array([p[1][0] for p in pts])
            sd = np.array([p[1][1] for p in pts])
            ax.fill_between(ks, mu - sd, mu + sd, color=col, alpha=0.28,
                            linewidth=0, zorder=1)
            ax.plot(ks, mu, ls, color=col, marker="o", ms=5.5,
                    lw=2.6 if t == OURS else 1.8, label=lab, zorder=3)
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
    fig.savefig(os.path.join(FIG, "fig_fewshot_curves.pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_datasets(data, metric="f1"):
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.2))
    for j, (ds, ax) in enumerate(zip(DNAME, axes.flat)):
        for t, (lab, col, ls) in PLOT.items():
            pts = [(k, mat(data, t, k, metric)) for k in KS]
            pts = [(k, a) for k, a in pts if a is not None]
            if not pts:
                continue
            ks = np.array([p[0] for p in pts])
            mu = np.array([p[1][:, j].mean() for p in pts])
            sd = np.array([p[1][:, j].std() for p in pts])
            ax.errorbar(ks, mu, yerr=sd, fmt=ls, color=col, marker="o", ms=2.5, capsize=1.5,
                        lw=1.8 if t == OURS else 1.0, label=lab)
        ax.set_title(ds, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.supxlabel("Number of samples per class", fontsize=10)
    fig.supylabel("Macro-F1 (%)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.legend(
        h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False,
        handlelength=1.5, handletextpad=0.35, columnspacing=0.9,
        bbox_to_anchor=(0.5, 0.952),
    )
    fig.savefig(os.path.join(FIG, "fig_fewshot_datasets.pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    data = load_all()
    summary = {"seeds": {t: {k: sorted(v.keys()) for k, v in d.items()} for t, d in data.items()},
               "avg": {}}
    for t in data:
        for k in KS:
            for m in METRICS:
                v = seed_avg(data, t, k, m)
                if v:
                    summary["avg"].setdefault(t, {}).setdefault(str(k), {})[m] = v
    for k in KS:
        s = wide_table(data, k, MAIN)
        if s:
            open(os.path.join(TAB, "fewshot_k%d.tex" % k), "w").write(s)
        e = wide_table(data, k, EXTRA, refs=False)
        if e:
            open(os.path.join(TAB, "fewshot_extra_k%d.tex" % k), "w").write(e)
    open(os.path.join(TAB, "fewshot_avg.tex"), "w").write(avg_table(data, MAIN))
    open(os.path.join(TAB, "fewshot_extra.tex"), "w").write(avg_table(data, EXTRA))
    open(os.path.join(TAB, "wilcoxon.tex"), "w").write(wilcoxon_table(data, summary))
    open(os.path.join(TAB, "efficiency.tex"), "w").write(efficiency_table(data))
    write_numbers(data, summary)
    lt, lm = level_table(data)
    open(os.path.join(TAB, "levels.tex"), "w").write(lt)
    open(os.path.join(TAB, "numbers.tex"), "a").write(lm)
    global SEEDS
    SEEDS = {0}
    open(os.path.join(TAB, "ablation.tex"), "w").write(avg_table(data, ABL, metrics=("acc", "f1")))
    open(os.path.join(TAB, "angle.tex"), "w").write(angle_table(data))
    dt, counts = dev_table(data)
    open(os.path.join(TAB, "dev_select.tex"), "w").write(dt)
    print("LOO selections:", counts)
    write_numbers(data, {}, prefix="z", fname="numbers_s0.tex")
    SEEDS = None
    fig4 = load_fig4()
    if fig4 is not None:
        rt, rm = reported_fewshot_table(data, fig4)
        open(os.path.join(TAB, "reported_fewshot.tex"), "w").write(rt)
        open(os.path.join(TAB, "numbers.tex"), "a").write(rm)
        plot_sota_f1(data, fig4)
    plot_curves(data)
    plot_datasets(data)
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    for t in summary["avg"]:
        print("%-12s" % t, "  ".join("k%s %.1f/%.1f(n=%d)" % (k, v["acc"][0], v["f1"][0], v["acc"][2])
                                     for k, v in summary["avg"][t].items()))


if __name__ == "__main__":
    main()
