#!/usr/bin/env python3
"""Few-shot wrappers for UniTS, LIMU-BERT-X and oneHAR/CDNet.

All classifiers take UniMTS acceleration windows ``(B, T, 66)`` (22 joints × 3)
and return class logits ``(B, ncls)``. Pretrained backbones live under
``$ROFT_THIRD_PARTY`` (default ``./third_party``); dataset-specific heads are randomly initialized.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import importlib.util
import math
import os
import sys
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = os.environ.get("ROFT_THIRD_PARTY", os.path.join(_REPO, "third_party"))
CKPT_DIR = os.path.join(ROOT, "ckpts")
LIMU_CKPT = os.path.join(ROOT, "LIMU-BERT-X", "weights", "limu_bert_x.pt")
UNITS_CKPT = os.path.join(CKPT_DIR, "units_x64_supervised_checkpoint.pth")
ONEHAR_CKPT = os.path.join(CKPT_DIR, "uniHAR.pth")
N_JOINTS = 22
LimuCfg = namedtuple(
    "LimuCfg", "hidden hidden_ff feature_num n_layers n_heads seq_len emb_norm"
)
LIMU_V4 = LimuCfg(72, 144, 6, 4, 4, 20, True)


def _load_file(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_matching(module, state, prefix=""):
    """Copy tensors whose name and shape match ``module``."""
    if prefix:
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    dst = module.state_dict()
    matched = {k: v for k, v in state.items() if k in dst and tuple(dst[k].shape) == tuple(v.shape)}
    module.load_state_dict(matched, strict=False)
    return len(matched), len(dst)


def _resample(x, t_out):
    """x: (..., T, C) -> same layout with T = t_out."""
    if x.size(-2) == t_out:
        return x
    lead = x.shape[:-2]
    t, c = x.shape[-2:]
    y = x.reshape(-1, t, c).transpose(1, 2)
    y = F.interpolate(y, size=t_out, mode="linear", align_corners=False)
    return y.transpose(1, 2).reshape(*lead, t_out, c)


def joints(x):
    b, t, d = x.shape
    if d != N_JOINTS * 3:
        raise ValueError("expected 66-D acc, got %s" % (tuple(x.shape),))
    return x.reshape(b, t, N_JOINTS, 3)


def active_mask(j):
    return j.abs().sum(dim=(1, 3)) > 1e-8


def mean_active_acc(x):
    j = joints(x)
    m = active_mask(j).to(dtype=j.dtype).view(j.size(0), 1, N_JOINTS, 1)
    return (j * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0)


def to_limu6(x, seq_len=20):
    acc = mean_active_acc(x)
    return _resample(torch.cat([acc, torch.zeros_like(acc)], dim=-1), seq_len)


def to_sensors(x, seq_len=120):
    return _resample(joints(x).permute(0, 2, 1, 3).contiguous(), seq_len)


# ---- LIMU-BERT Transformer (parameter-sharing, copied from LIMU-BERT-X/models.py) ----
def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _split_last(x, shape):
    shape = list(shape)
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -int(np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def _merge_last(x, n_dims):
    s = x.size()
    return x.view(*s[:-n_dims], -1)


class _LayerNorm(nn.Module):
    def __init__(self, cfg, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta = nn.Parameter(torch.zeros(cfg.hidden))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class _Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)
        self.norm = _LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        return self.norm(e + self.pos_embed(pos))


class _MHSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        self.n_heads = cfg.n_heads

    def forward(self, x):
        qkv = (self.proj_q(x), self.proj_k(x), self.proj_v(x))
        q, k, v = (_split_last(t, (self.n_heads, -1)).transpose(1, 2) for t in qkv)
        scores = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.size(-1)), dim=-1)
        return _merge_last((scores @ v).transpose(1, 2).contiguous(), 2)


class _PWFF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        return self.fc2(_gelu(self.fc1(x)))


class LimuTransformer(nn.Module):
    def __init__(self, cfg=LIMU_V4):
        super().__init__()
        self.embed = _Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = _MHSA(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = _LayerNorm(cfg)
        self.pwff = _PWFF(cfg)
        self.norm2 = _LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


def _load_limu(encoder, ckpt=LIMU_CKPT):
    raw = torch.load(ckpt, map_location="cpu")
    n, tot = _load_matching(encoder, raw, prefix="transformer.")
    if n < 8:
        raise RuntimeError("LIMU-BERT-X load failed (%d/%d) from %s" % (n, tot, ckpt))
    return n


class LIMUBERTXClassifier(nn.Module):
    """LIMU-BERT-X encoder (MobiCom'25 weights) + linear classifier.

    Mean-pools active joints, pads gyro with zeros (UniMTS is acc-only),
    resamples to the pretrained seq_len=20.
    """

    def __init__(self, ncls, ckpt=LIMU_CKPT):
        super().__init__()
        self.cfg = LIMU_V4
        self.encoder = LimuTransformer(self.cfg)
        self.n_loaded = _load_limu(self.encoder, ckpt)
        self.head = nn.Linear(self.cfg.hidden, ncls)

    def encode(self, x):
        return self.encoder(to_limu6(x, self.cfg.seq_len)).mean(dim=1)

    def forward(self, x):
        return self.head(self.encode(x))


class UniTSClassifier(nn.Module):
    """UniTS x64 backbone (NeurIPS 2024) + dataset-specific linear head.

    Multivariate input keeps all 66 acceleration channels. Prompt tokens are
    created for this dataset; shared blocks / patch embed / CLS head load from
    the official supervised x64 checkpoint.
    """

    def __init__(self, ncls, ckpt=UNITS_CKPT, seq_len=200, enc_in=66):
        super().__init__()
        try:
            import timm  # noqa: F401
        except ImportError as exc:
            raise ImportError("UniTS requires timm: pip install 'timm>=0.9'") from exc
        uni = _load_file("units_backbone_mod", os.path.join(ROOT, "UniTS", "models", "UniTS.py"))
        args = Namespace(
            prompt_num=10, patch_len=16, stride=16, e_layers=3, d_model=64,
            n_heads=8, dropout=0.1,
        )
        configs = [("CLS_UniMTS", {
            "task_name": "classification",
            "dataset": "UniMTS",
            "enc_in": enc_in,
            "num_class": int(ncls),
            "seq_len": int(seq_len),
        })]
        self.units = uni.Model(args, configs, pretrain=False)
        self.n_loaded = 0
        if os.path.isfile(ckpt) and os.path.getsize(ckpt) >= 1_000_000:
            raw = torch.load(ckpt, map_location="cpu")
            if isinstance(raw, dict) and "student" in raw:
                raw = raw["student"]
            elif isinstance(raw, dict) and isinstance(raw.get("model"), dict):
                raw = raw["model"]
            raw = {k.replace("module.", ""): v for k, v in raw.items()}
            self.n_loaded, _ = _load_matching(self.units, raw)
        self.seq_len = seq_len
        self.head = nn.Linear(args.d_model, ncls)

    def _tokens(self, x):
        m = self.units
        dname = m.configs_list[0][1]["dataset"]
        tname = m.configs_list[0][0]
        tokens, _, _, n_vars, _ = m.tokenize(x)
        seq_len = tokens.shape[-2]
        tokens = m.prepare_prompt(
            tokens, n_vars, m.prompt_tokens[dname], m.cls_tokens[tname], 1,
            task_name="classification",
        )
        return m.backbone(tokens, m.prompt_tokens[dname].shape[2], seq_len)

    def encode(self, x):
        x = _resample(x, self.seq_len)
        feat = self.units.cls_head(self._tokens(x), return_feature=True)
        return feat.mean(dim=(1, 2))

    def forward(self, x):
        return self.head(self.encode(x))


class CDNetClassifier(nn.Module):
    """Vectorized CDNet (IMWUT 2025 oneHAR) for UniMTS 22-joint windows.

    Official HARModel.forward calls LIMU via numpy per sensor. This port keeps
    the ViT-per-axis + GRU fusion, uses LIMU-BERT-X as the IMU encoder, and
    interpolates positional embeddings so N can exceed the original 5 locations.
    ``uniHAR.pth`` tensors are loaded when the real (non-LFS) file is present.
    """

    def __init__(self, ncls, limu_ckpt=LIMU_CKPT, cdnet_ckpt=ONEHAR_CKPT, seq_len=120):
        super().__init__()
        self.seq_len = seq_len
        self.limu = LimuTransformer(LIMU_V4)
        self.n_loaded = _load_limu(self.limu, limu_ckpt)
        self.limu_len = LIMU_V4.seq_len
        self.hidden = LIMU_V4.hidden

        vit = _load_file("unihar_vit_mod", os.path.join(ROOT, "uniHAR", "VITmodel.py"))
        self.input_ln = nn.LayerNorm([seq_len, 3])
        self.vit_x = vit.VisionTransformer(embed_dim=seq_len)
        self.vit_y = vit.VisionTransformer(embed_dim=seq_len)
        self.vit_z = vit.VisionTransformer(embed_dim=seq_len)
        self.gru1 = nn.GRU(3, 9, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(18, 36, num_layers=1, batch_first=True, bidirectional=True)
        self.out_ln = nn.LayerNorm([seq_len, self.hidden])
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=0)
        self.fc1 = nn.Linear((seq_len - 6) * (self.hidden - 6), 768)
        self.head = nn.Linear(768, ncls)
        if os.path.isfile(cdnet_ckpt) and os.path.getsize(cdnet_ckpt) > 1_000_000:
            raw = torch.load(cdnet_ckpt, map_location="cpu")
            renamed = {}
            for k, v in raw.items():
                nk = k
                for p in ("imu_encoder.har_model.", "har_model.", "imu_encoder."):
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        break
                renamed[nk] = v
            extra, _ = _load_matching(self, renamed)
            self.n_loaded += extra

    def _limu_branch(self, sensors, mask):
        b, n, _, _ = sensors.shape
        acc = _resample(sensors, self.limu_len)
        x6 = torch.cat([acc, torch.zeros_like(acc)], dim=-1).reshape(b * n, self.limu_len, 6)
        h = self.limu(x6).reshape(b, n, self.limu_len, self.hidden)
        w = mask.to(dtype=h.dtype).view(b, n, 1, 1)
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return _resample(pooled, self.seq_len).unsqueeze(1)

    def _vit_one(self, vit, x):
        b, n, _ = x.shape
        tok = torch.cat((vit.cls_token.expand(b, -1, -1), x), dim=1)
        pe = vit.pos_embed
        need = n + 1
        if pe.size(1) != need:
            pe = F.interpolate(pe.transpose(1, 2), size=need, mode="linear", align_corners=False).transpose(1, 2)
        return vit.norm(vit.blocks(vit.pos_drop(tok + pe.to(dtype=tok.dtype))))[:, -1].unsqueeze(1)

    def _vit_branch(self, sensors):
        b, n, t, c = sensors.shape
        x = self.input_ln(sensors.reshape(b * n, t, c)).reshape(b, n, t, c)
        fused = torch.cat([
            self._vit_one(self.vit_x, x[..., 0]),
            self._vit_one(self.vit_y, x[..., 1]),
            self._vit_one(self.vit_z, x[..., 2]),
        ], dim=1).permute(0, 2, 1)
        gru, _ = self.gru1(fused)
        gru, _ = self.gru2(gru)
        return gru.unsqueeze(1)

    def encode(self, x):
        sensors = to_sensors(x, self.seq_len)
        fused = torch.cat([self._limu_branch(sensors, active_mask(joints(x))), self._vit_branch(sensors)], dim=1)
        fused = self.out_ln(fused.reshape(-1, self.seq_len, self.hidden)).reshape(
            x.size(0), 2, self.seq_len, self.hidden
        )
        return self.fc1(self.conv3(self.conv2(self.conv1(fused))).flatten(1))

    def forward(self, x):
        return self.head(self.encode(x))


def build(name, ncls, **kwargs):
    name = name.lower().replace("-", "_")
    if name in ("limu", "limu_bert", "limu_bert_x", "limubertx"):
        return LIMUBERTXClassifier(ncls, **kwargs)
    if name in ("units", "units_x64"):
        return UniTSClassifier(ncls, **kwargs)
    if name in ("onehar", "cdnet", "unihar"):
        return CDNetClassifier(ncls, **kwargs)
    from ext_sota import build as build_sota
    return build_sota(name, ncls, **kwargs)


NAMES = ("units", "limu", "onehar", "biobank", "gpt4ts", "imagebind")
