#!/usr/bin/env python3
"""Unified few-shot benchmark under the UniMTS protocol (RoFT entry point).

18 real datasets, official few_shot_data_2 support indices, k in {1,2,3,5,10}.
Metrics: Acc, macro-F1, R@2 (same definitions as UniMTS evaluate.py / finetune.py).

RoFT (paper default)
  python tools/fewshot_bench.py --method unimts_ft --tag ft_small_ens \\
      --aug small --aug_angle 30 --ens 1 --k 1 --seed 0

Methods
  unimts_ft  fine-tune graph encoder + linear head (CE, Adam 1e-4, lowest support loss).
             Combine with --aug small --ens 1 for RoFT.
  probe      linear probe on frozen encoder.
  frozen     zero-shot / 1-NN / prototype / Tip-Adapter / Tip-Adapter-F.
  tipguided  optional fused-logit ablation (text + Tip + CE head).
  units/limu/onehar/biobank/gpt4ts/imagebind  optional external backbones.
  random     ST-GCN trained from scratch (no UniMTS checkpoint).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import random
import time

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.contrastive import ContrastiveModule
from utils.metrics import compute_metrics_np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = [
    "Opp_g", "UCIHAR", "MotionSense", "w-HAR", "Shoaib", "har70plus",
    "realworld", "TNDA-HAR", "PAMAP", "USCHAD", "Mhealth", "Harth",
    "ut-complex", "Wharf", "WISDM", "DSADS", "UTD-MHAD", "MMAct",
]
TEMPLATES = ["{}", "a person is {}", "the activity is {}", "someone is {}"]
CACHE_DIR = os.environ.get("ROFT_CACHE", os.path.join(ROOT, "input_cache"))
FROZEN_METHODS = ["zeroshot", "nn1", "proto", "tip", "tip_f"]
ACC_IDX = np.array([range(i, i + 3) for i in range(0, 132, 6)]).flatten()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cached_load(ds, split, k, data_path, padding_size=200):
    """Acceleration-only inputs (N, T, 66), labels, label names, num_classes."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = "k%d" % k if split == "train" else "all"
    path = os.path.join(CACHE_DIR, "%s_%s_%s.pt" % (ds, split, tag))
    if os.path.exists(path):
        d = torch.load(path)
        return d["x"], d["y"], d["names"], d["ncls"]
    from datasets.data import load

    x, _, y, names, _, ncls = load(
        ds, padding_size, data_path, split=split, k=k if split == "train" else None
    )
    x = x[:, :, ACC_IDX].contiguous()
    d = {"x": x, "y": y.long(), "names": list(names), "ncls": int(ncls)}
    tmp = path + ".tmp%d" % os.getpid()
    torch.save(d, tmp)
    os.replace(tmp, path)
    return d["x"], d["y"], d["names"], d["ncls"]


def to_graph(xb):
    xb = xb.cuda(non_blocking=True)
    b, t, _ = xb.shape
    return xb.reshape(b, t, 22, 3).permute(0, 3, 1, 2).unsqueeze(-1)


def score(y_true, pred, logits, y_rank):
    """y_rank: labels expressed as column indices of `logits`."""
    _, r2, _, _, _, mrr = compute_metrics_np(logits, y_rank)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, average="macro")),
        "R@2": float(r2),
        "MRR": float(mrr),
    }


@torch.no_grad()
def encode_text_ensemble(model, names):
    acc = None
    for tmpl in TEMPLATES:
        tokens = clip.tokenize([tmpl.format(n) for n in names], truncate=True).cuda()
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        acc = f if acc is None else acc + f
    acc = acc / len(TEMPLATES)
    return acc / acc.norm(dim=-1, keepdim=True)


def l2n(f):
    return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_rotations(n, max_angle=None, device="cuda"):
    """(n, 3, 3) rotations: uniform over SO(3) if max_angle is None, else a random axis
    with angle ~ U(-max_angle, max_angle)."""
    if max_angle is None:
        q = F.normalize(torch.randn(n, 4, device=device), dim=-1)
        w, x, y, z = q.unbind(-1)
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], -1).view(n, 3, 3)
    axis = F.normalize(torch.randn(n, 3, device=device), dim=-1)
    ang = (torch.rand(n, device=device) * 2 - 1) * max_angle
    kx = torch.zeros(n, 3, 3, device=device)
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    kx = kx - kx.transpose(1, 2)
    s, c = ang.sin()[:, None, None], ang.cos()[:, None, None]
    return torch.eye(3, device=device) + s * kx + (1 - c) * (kx @ kx)


def prob_ensemble(logits_a, logits_b):
    """log of the equal-weight average of two softmax distributions (output-space ensemble)."""
    pa = torch.softmax(torch.from_numpy(logits_a).double(), -1)
    pb = torch.softmax(torch.from_numpy(logits_b).double(), -1)
    return torch.log(0.5 * pa + 0.5 * pb).numpy()


def augment(g, mode, max_deg=30.0):
    """Rotation augmentation of graph input (B, 3, T, V, 1): one rotation per sample and joint,
    shared over time, as in UniMTS pre-training. Zero-padded joints stay zero."""
    if mode == "none":
        return g
    b, _, _, v, _ = g.shape
    rot = random_rotations(b * v, None if mode == "so3" else float(np.deg2rad(max_deg)), g.device).view(b, v, 3, 3)
    return torch.einsum("bvcd,bdtv->bctv", rot, g[..., 0]).unsqueeze(-1)


EXT_METHODS = ("units", "limu", "onehar", "biobank", "gpt4ts", "imagebind")


def acc_from_graph(g):
    """(B, 3, T, V, 1) -> (B, T, 66)."""
    b, _, t, v, _ = g.shape
    return g.squeeze(-1).permute(0, 2, 3, 1).reshape(b, t, v * 3)


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.ckpt = None
        if args.method in EXT_METHODS:
            return
        args.stage = "finetune"
        args.gyro = 0
        args.stft = 0
        args.num_class = 2
        self.model = ContrastiveModule(args).cuda()
        self.ckpt = None if args.method == "random" else torch.load(args.checkpoint, map_location="cpu")

    @torch.no_grad()
    def dump(self, ds, m, te_x, te_y):
        """Save normalized test embeddings when --dump_feats is set (for t-SNE)."""
        if not self.args.dump_feats:
            return
        feats = torch.cat([l2n(m.encode_image(to_graph(xb))).float().cpu()
                           for (xb,) in DataLoader(TensorDataset(te_x), batch_size=self.args.batch_size)])
        os.makedirs(self.args.dump_feats, exist_ok=True)
        np.savez(os.path.join(self.args.dump_feats, "%s_%s_k%d_s%d.npz" % (self.args.tag, ds, self.args.k, self.args.seed)),
                 feat=feats.numpy(), y=te_y.numpy())

    def reset(self, n_out):
        m = self.model
        if self.args.method == "random":
            from models.model import ST_GCN_18
            m.model.acc = ST_GCN_18(in_channels=3).float().cuda()
        else:
            m.model.load_state_dict(self.ckpt)
        m.fc = nn.Linear(512, n_out).cuda()
        return m

    # ---------------------------------------------------------------- official
    def run_official(self, ds, tr_x, tr_y, te_x, te_y, ncls, mode, names=None):
        a = self.args
        m = self.reset(ncls)
        zs_logits = None
        if a.ens and names is not None:
            m.eval()
            text = encode_text_ensemble(m, [names[c] if c < len(names) else str(c) for c in range(ncls)])
            tau = m.model.logit_scale.exp().detach()
            with torch.no_grad():
                zs_logits = torch.cat([(tau * l2n(m.encode_image(to_graph(xb))) @ text.t()).float().cpu()
                                       for (xb,) in DataLoader(TensorDataset(te_x), batch_size=a.batch_size)]).numpy()
        if mode == "probe":
            for p in m.model.parameters():
                p.requires_grad = False
        else:
            for p in m.model.parameters():
                p.requires_grad = True
        params = [p for p in m.parameters() if p.requires_grad]
        n_params = sum(p.numel() for n, p in m.named_parameters()
                       if p.requires_grad and ("model.acc" in n or n.startswith("fc.")))
        opt = torch.optim.Adam(params, lr=a.lr)
        loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=a.batch_size, shuffle=True)

        best_loss, best = None, None
        t0 = time.time()
        for _ in range(a.epochs):
            m.train()
            tol = 0.0
            for xb, yb in loader:
                yb = yb.cuda()
                loss = F.cross_entropy(m.classifier(augment(to_graph(xb), a.aug, a.aug_angle)).float(), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tol += len(yb) * loss.item()
            if best_loss is None or tol < best_loss:
                best_loss = tol
                best = {
                    "acc": {k: v.detach().clone() for k, v in m.model.acc.state_dict().items()},
                    "fc": {k: v.detach().clone() for k, v in m.fc.state_dict().items()},
                }
        train_s = time.time() - t0
        m.model.acc.load_state_dict(best["acc"])
        m.fc.load_state_dict(best["fc"])
        m.eval()
        self.dump(ds, m, te_x, te_y)
        logits = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(te_x), batch_size=a.batch_size):
                logits.append(m.classifier(to_graph(xb)).float().cpu())
        logits = torch.cat(logits).numpy()
        y = te_y.numpy()
        res = score(y, logits.argmax(1), logits, y)
        if zs_logits is not None:
            ens = prob_ensemble(logits, zs_logits)
            res["ens"] = score(y, ens.argmax(1), ens, y)
        res.update({"train_s": train_s, "n_params": int(n_params)})
        return res

    # ---------------------------------------------------------------- UniTS / LIMU-BERT-X / oneHAR
    def run_external(self, ds, tr_x, tr_y, te_x, te_y, ncls):
        from tools.baselines.ext_backbones import build

        a = self.args
        m = build(a.method, ncls).cuda()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=min(a.batch_size, max(1, len(tr_x))), shuffle=True)
        best_loss, best = None, None
        t0 = time.time()
        for _ in range(a.epochs):
            m.train()
            tol = 0.0
            for xb, yb in loader:
                yb = yb.cuda()
                xb = acc_from_graph(augment(to_graph(xb), a.aug, a.aug_angle))
                loss = F.cross_entropy(m(xb).float(), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tol += len(yb) * loss.item()
            if best_loss is None or tol < best_loss:
                best_loss = tol
                best = {k: v.detach().clone() for k, v in m.state_dict().items()}
        train_s = time.time() - t0
        m.load_state_dict(best)
        m.eval()
        logits = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(te_x), batch_size=a.batch_size):
                logits.append(m(xb.cuda(non_blocking=True)).float().cpu())
        logits = torch.cat(logits).numpy()
        y = te_y.numpy()
        res = score(y, logits.argmax(1), logits, y)
        res.update({"train_s": train_s, "n_params": int(n_params), "n_loaded": int(getattr(m, "n_loaded", 0))})
        return res

    # ---------------------------------------------------------------- tipguided
    def run_tipguided(self, ds, tr_x, tr_y, te_x, te_y, names, beta, alpha):
        a = self.args
        class_ids = sorted(set(int(v) for v in tr_y.numpy()))
        mapping = {c: i for i, c in enumerate(class_ids)}
        inv = np.array(class_ids)
        n_c = len(class_ids)
        y_c = torch.tensor([mapping[int(v)] for v in tr_y.numpy()], dtype=torch.long)

        m = self.reset(n_c)
        for n, p in m.named_parameters():
            p.requires_grad = ("model.acc" in n) or n.startswith("fc.")
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

        text = encode_text_ensemble(m, [names[c] if c < len(names) else str(c) for c in class_ids])
        tau = m.model.logit_scale.exp().detach()

        m.eval()
        with torch.no_grad():
            keys0 = torch.cat([l2n(m.encode_image(to_graph(xb)))
                               for (xb,) in DataLoader(TensorDataset(tr_x), batch_size=a.batch_size)])
        values = F.one_hot(y_c.cuda(), num_classes=n_c).float()
        values = values / values.sum(0, keepdim=True).clamp_min(1.0)

        keys = nn.Parameter(keys0.clone())
        log_a = nn.Parameter(torch.tensor(float(np.log(alpha)), device="cuda"))
        log_g = nn.Parameter(torch.tensor(0.0, device="cuda"))
        cache_params = []
        if a.use_tip:
            cache_params += [keys, log_a]
        if a.use_ce and (a.use_tip or a.use_text):
            cache_params += [log_g]
        head_lr = a.head_lr if a.head_lr > 0 else a.lr
        groups = [{"params": [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("fc.")],
                   "lr": a.lr},
                  {"params": list(m.fc.parameters()), "lr": head_lr}]
        if cache_params:
            groups.append({"params": cache_params, "lr": a.lr_cache})
        opt = torch.optim.Adam(groups)

        def forward(feat):
            prior = None
            if a.use_text:
                prior = tau * (feat @ text.t())
            if a.use_tip:
                aff = torch.exp(-beta * (1.0 - feat @ l2n(keys).t()))
                t = torch.exp(log_a) * (aff @ values)
                prior = t if prior is None else prior + t
            ce = m.fc(feat) if a.use_ce else None
            if ce is None:
                return prior, prior, None
            if prior is None:
                return ce, None, ce
            return prior + torch.exp(log_g) * ce, prior, ce

        init_logits = None
        if a.ens:
            m.eval()
            with torch.no_grad():
                init_logits = torch.cat([forward(l2n(m.encode_image(to_graph(xb))))[0].float().cpu()
                                         for (xb,) in DataLoader(TensorDataset(te_x), batch_size=a.batch_size)]).numpy()

        loader = DataLoader(TensorDataset(tr_x, y_c),
                            batch_size=min(a.batch_size, max(4, len(tr_x))), shuffle=True)
        best_tr, best = -1.0, None
        t0 = time.time()
        for _ in range(a.epochs):
            m.train()
            for xb, yb in loader:
                yb = yb.cuda()
                logits, prior, ce = forward(l2n(m.encode_image(augment(to_graph(xb), a.aug, a.aug_angle))))
                loss = F.cross_entropy(logits, yb)
                if a.kl_w > 0 and prior is not None and ce is not None:
                    loss = loss + a.kl_w * F.kl_div(
                        F.log_softmax(ce, -1), F.softmax(prior.detach(), -1), reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                correct = 0
                for xb, yb in DataLoader(TensorDataset(tr_x, y_c), batch_size=a.batch_size):
                    logits, _, _ = forward(l2n(m.encode_image(to_graph(xb))))
                    correct += (logits.argmax(1).cpu() == yb).sum().item()
                tr_acc = correct / len(y_c)
            if tr_acc >= best_tr:
                best_tr = tr_acc
                best = {
                    "acc": {k: v.detach().clone() for k, v in m.model.acc.state_dict().items()},
                    "fc": {k: v.detach().clone() for k, v in m.fc.state_dict().items()},
                    "keys": keys.detach().clone(),
                    "log_a": log_a.detach().clone(),
                    "log_g": log_g.detach().clone(),
                }
        train_s = time.time() - t0
        m.model.acc.load_state_dict(best["acc"])
        m.fc.load_state_dict(best["fc"])
        with torch.no_grad():
            keys.copy_(best["keys"])
            log_a.copy_(best["log_a"])
            log_g.copy_(best["log_g"])
        m.eval()
        self.dump(ds, m, te_x, te_y)
        logits = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(te_x), batch_size=a.batch_size):
                logits.append(forward(l2n(m.encode_image(to_graph(xb))))[0].float().cpu())
        logits = torch.cat(logits).numpy()
        y = te_y.numpy()
        pred = inv[logits.argmax(1)]
        known = np.array([int(v) in mapping for v in y])
        y_rank = np.array([mapping[int(v)] for v in y[known]])
        res = score(y, pred, logits[known], y_rank)
        if init_logits is not None:
            ens = prob_ensemble(logits, init_logits)
            res["ens"] = score(y, inv[ens.argmax(1)], ens[known], y_rank)
        res.update({
            "train_s": train_s, "n_params": int(n_params), "best_tr": best_tr,
            "alpha": float(best["log_a"].exp()), "gamma": float(best["log_g"].exp()),
            "beta": float(beta),
        })
        return res


    # ---------------------------------------------------------------- frozen encoder
    @torch.no_grad()
    def _encode(self, m, x):
        return torch.cat([l2n(m.encode_image(to_graph(xb))).float()
                          for (xb,) in DataLoader(TensorDataset(x), batch_size=self.args.batch_size)])

    def run_frozen(self, ds, tr_x, tr_y, te_x, te_y, names, beta, alpha, tipf_epochs=40):
        """Zero-shot, 1-NN, prototype, Tip-Adapter and Tip-Adapter-F on frozen features."""
        class_ids = sorted(set(int(v) for v in tr_y.numpy()))
        mapping = {c: i for i, c in enumerate(class_ids)}
        inv = np.array(class_ids)
        n_c = len(class_ids)
        y_c = torch.tensor([mapping[int(v)] for v in tr_y.numpy()], dtype=torch.long).cuda()

        m = self.reset(n_c)
        m.eval()
        self.dump(ds, m, te_x, te_y)
        t0 = time.time()
        text = encode_text_ensemble(m, [names[c] if c < len(names) else str(c) for c in class_ids]).float()
        tau = float(m.model.logit_scale.exp())
        sf = self._encode(m, tr_x)
        qf = self._encode(m, te_x)
        enc_s = time.time() - t0

        y = te_y.numpy()
        known = np.array([int(v) in mapping for v in y])
        y_rank = np.array([mapping[int(v)] for v in y[known]])

        def finish(logits, extra_s=0.0, n_params=0):
            logits = logits.float().cpu().numpy()
            r = score(y, inv[logits.argmax(1)], logits[known], y_rank)
            r.update({"train_s": extra_s, "n_params": int(n_params), "beta": float(beta), "alpha": float(alpha)})
            return r

        values = F.one_hot(y_c, num_classes=n_c).float()
        values = values / values.sum(0, keepdim=True).clamp_min(1.0)
        zs_q = tau * (qf @ text.t())

        out = {"zeroshot": finish(zs_q)}
        sim = qf @ sf.t()
        out["nn1"] = finish(torch.stack([sim[:, y_c == c].amax(1) for c in range(n_c)], 1))
        protos = l2n(values.t() @ sf)
        out["proto"] = finish(qf @ protos.t())

        def tip(keys, a):
            return zs_q + a * (torch.exp(-beta * (1.0 - qf @ l2n(keys).t())) @ values)

        out["tip"] = finish(tip(sf, alpha))

        t1 = time.time()
        keys = nn.Parameter(sf.clone())
        opt = torch.optim.Adam([keys], lr=self.args.lr_cache)
        zs_s = tau * (sf @ text.t())
        for _ in range(tipf_epochs):
            logits = zs_s + alpha * (torch.exp(-beta * (1.0 - sf @ l2n(keys).t())) @ values)
            loss = F.cross_entropy(logits, y_c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out["tip_f"] = finish(tip(keys, alpha), time.time() - t1, keys.numel())
        for r in out.values():
            r["encode_s"] = enc_s
        return out


def loo_hyp(k):
    path = os.environ.get(
        "ROFT_TIP_LOO",
        os.path.join(ROOT, "logs", "cross_tip_k%d.json" % k),
    )
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return {r["dataset"]: (r["loo_hyp_tip_f"]["beta"], r["loo_hyp_tip_f"]["alpha"]) for r in d["rows"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["unimts_ft", "probe", "tipguided", "frozen", "units", "limu", "onehar",
                                        "biobank", "gpt4ts", "imagebind", "random"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_cache", type=float, default=1e-3)
    p.add_argument("--kl_w", type=float, default=0.1)
    p.add_argument("--use_text", type=int, default=1)
    p.add_argument("--use_tip", type=int, default=1)
    p.add_argument("--use_ce", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--padding_size", type=int, default=200)
    p.add_argument("--data_path", default="./UniMTS_data/")
    p.add_argument("--checkpoint", default="./checkpoint/UniMTS.pth")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--out_dir", default="./logs/bench")
    p.add_argument("--warm_cache", action="store_true")
    p.add_argument("--dump_feats", default="")
    p.add_argument("--aug", choices=["none", "so3", "small"], default="none")
    p.add_argument("--aug_angle", type=float, default=30.0, help="max rotation angle (deg) for --aug small")
    p.add_argument("--head_lr", type=float, default=0.0, help="tipguided head lr (0: same as --lr)")
    p.add_argument("--ens", type=int, default=1,
                   help="also report the equal-weight output ensemble with the initial (prior) predictor")
    a = p.parse_args()
    if a.epochs is None:
        a.epochs = 200 if a.method in ("unimts_ft", "probe", "random") + EXT_METHODS else 100
    datasets = a.datasets.split(",")

    if a.warm_cache:
        for ds in datasets:
            cached_load(ds, "test", None, a.data_path, a.padding_size)
            for k in (1, 2, 3, 5, 10):
                cached_load(ds, "train", k, a.data_path, a.padding_size)
            print("cached", ds, flush=True)
        return

    os.makedirs(a.out_dir, exist_ok=True)
    subs = FROZEN_METHODS if a.method == "frozen" else [a.tag]
    outs = {s: os.path.join(a.out_dir, "%s_k%d_s%d.json" % (s, a.k, a.seed)) for s in subs}
    if all(os.path.exists(o) for o in outs.values()):
        print("exists", list(outs.values()))
        return
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
    hyp = loo_hyp(a.k) if a.method in ("tipguided", "frozen") else {}
    runner = Runner(a)
    rows = {s: [] for s in subs}
    print("%s k=%d seed=%d" % (a.tag, a.k, a.seed), flush=True)
    for ds in datasets:
        set_seed(a.seed)
        tr_x, tr_y, names, ncls = cached_load(ds, "train", a.k, a.data_path, a.padding_size)
        te_x, te_y, _, _ = cached_load(ds, "test", None, a.data_path, a.padding_size)
        if a.method == "frozen":
            res_ds = runner.run_frozen(ds, tr_x, tr_y, te_x, te_y, names, *hyp[ds])
        elif a.method == "tipguided":
            res_ds = {a.tag: runner.run_tipguided(ds, tr_x, tr_y, te_x, te_y, names, *hyp[ds])}
        elif a.method in EXT_METHODS:
            res_ds = {a.tag: runner.run_external(ds, tr_x, tr_y, te_x, te_y, ncls)}
        else:
            res_ds = {a.tag: runner.run_official(ds, tr_x, tr_y, te_x, te_y, ncls, a.method, names)}
        if "ens" in res_ds.get(a.tag, {}):
            ens = res_ds[a.tag].pop("ens")
            res_ds[a.tag + "_ens"] = {**ens, **{kk: vv for kk, vv in res_ds[a.tag].items() if kk not in ens}}
        for s, r in res_ds.items():
            rows.setdefault(s, [])
            rows[s].append({"dataset": ds, **r})
            print("%-12s %-10s acc=%.4f f1=%.4f R@2=%.4f (%.0fs)"
                  % (ds, s, r["acc"], r["f1"], r["R@2"], r["train_s"]), flush=True)
        torch.cuda.empty_cache()
    for s in rows:
        outs.setdefault(s, os.path.join(a.out_dir, "%s_k%d_s%d.json" % (s, a.k, a.seed)))
        avg = {m: float(np.mean([r[m] for r in rows[s]])) for m in ("acc", "f1", "R@2")}
        res = {"tag": s, "method": a.method, "k": a.k, "seed": a.seed,
               "config": {kk: vv for kk, vv in vars(a).items() if kk not in ("datasets",)},
               "average": avg, "rows": rows[s]}
        tmp = outs[s] + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, outs[s])
        print("AVERAGE", s, avg, "->", outs[s], flush=True)


if __name__ == "__main__":
    main()
