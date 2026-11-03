#!/usr/bin/env python3
"""Few-shot wrappers for UniMTS Figure-4 models that have public weights.

ImageBind (IMU encoder of imagebind_huge), BioBankSSL (OxWearables harnet10)
and GPT4TS (frozen GPT-2, 6 layers). All take UniMTS windows ``(B, T, 66)``
and return logits ``(B, ncls)``. IMU2CLIP and IMUGPT are omitted: the former
never released weights (UniMTS re-pretrained on Ego4D); the latter trains
DeepConvLSTM on GPT-generated IMU rather than a frozen pretrained encoder.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.baselines.ext_backbones import (
    CKPT_DIR, ROOT, _load_file, _load_matching, _resample, mean_active_acc,
)

IMAGEBIND_CKPT = os.path.join(CKPT_DIR, "imagebind_huge.pth")
IMAGEBIND_IMU = os.path.join(CKPT_DIR, "imagebind_imu.pth")
BIOBANK_DIR = os.path.join(ROOT, "ssl-wearables")
BIOBANK_CKPT = os.path.join(BIOBANK_DIR, "model_check_point", "mtl_best.mdl")
GPT2_NAME = os.environ.get("GPT2_NAME", "gpt2")
_IB_STATE = None
_GPT_TRUNK = None


def _imagebind_state(ckpt=IMAGEBIND_CKPT):
    """Load imagebind_huge once and cache IMU-only tensors (~tens of MB)."""
    global _IB_STATE
    if _IB_STATE is not None:
        return _IB_STATE
    if os.path.isfile(IMAGEBIND_IMU) and os.path.getsize(IMAGEBIND_IMU) > 1_000_000:
        _IB_STATE = torch.load(IMAGEBIND_IMU, map_location="cpu")
        return _IB_STATE
    if not os.path.isfile(ckpt) or os.path.getsize(ckpt) < 1_000_000_000:
        raise FileNotFoundError("ImageBind checkpoint missing: %s" % ckpt)
    raw = torch.load(ckpt, map_location="cpu")
    _IB_STATE = {k: v for k, v in raw.items() if ".imu" in k or k.startswith("imu")}
    tmp = IMAGEBIND_IMU + ".tmp"
    torch.save(_IB_STATE, tmp)
    os.replace(tmp, IMAGEBIND_IMU)
    del raw
    return _IB_STATE


def _gpt_trunk(gpt_layers=6):
    global _GPT_TRUNK
    if _GPT_TRUNK is None:
        os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
        from transformers import GPT2Model
        gpt = GPT2Model.from_pretrained(GPT2_NAME)
        gpt.h = nn.ModuleList(list(gpt.h)[:gpt_layers])
        _GPT_TRUNK = gpt
    import copy
    return copy.deepcopy(_GPT_TRUNK)


def to_wrist3(x, seq_len, gravity=True):
    """Mean-pool active joints to 3-axis acc; optional m/s^2 -> g."""
    acc = mean_active_acc(x)
    if gravity:
        acc = acc / 9.80665
    return _resample(acc, seq_len)


def to_imu6(x, seq_len=2000):
    """ImageBind IMU: 6 x T (acc + zero gyro) at the pretrained 200 Hz / 10 s."""
    acc = _resample(mean_active_acc(x), seq_len)
    return torch.cat([acc, torch.zeros_like(acc)], dim=-1).transpose(1, 2)


class _Transpose01(nn.Module):
    def forward(self, x):
        return x.transpose(0, 1)


class _SelectCLS(nn.Module):
    def forward(self, x, **kwargs):
        return x[:, 0]


class _MHA(nn.MultiheadAttention):
    def forward(self, x, attn_mask=None):
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class _Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class _Block(nn.Module):
    def __init__(self, dim, attn_target, mlp_ratio=4, ffn_dropout_rate=0.0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **_):
        super().__init__()
        self.attn = attn_target()
        self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(mlp_ratio * dim), ffn_dropout_rate)
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = None

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
        return x + self.drop_path(self.mlp(self.norm_2(x)))


class _SimpleTransformer(nn.Module):
    def __init__(self, attn_target, embed_dim, num_blocks, pre_transformer_layer=None,
                 post_transformer_layer=None, drop_path_rate=0.0, **_):
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        self.blocks = nn.Sequential(*[
            _Block(embed_dim, attn_target) for _ in range(num_blocks)
        ])
        self.post_transformer_layer = post_transformer_layer

    def forward(self, tokens, attn_mask=None, **_):
        if self.pre_transformer_layer is not None:
            tokens = self.pre_transformer_layer(tokens)
        for blk in self.blocks:
            tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer is not None:
            tokens = self.post_transformer_layer(tokens)
        return tokens


class _IMUPreprocessor(nn.Module):
    def __init__(self, embed_dim=512, kernel_size=8, img_size=(6, 2000)):
        super().__init__()
        self.kernel_size = kernel_size
        self.imu_stem = nn.Module()
        self.imu_stem.proj = nn.Linear(48, embed_dim, bias=False)
        self.imu_stem.norm_layer = nn.LayerNorm(embed_dim)
        n_tok = img_size[1] // kernel_size + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, n_tok, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.01)
        nn.init.normal_(self.cls_token, std=embed_dim ** -0.5)

    def forward(self, imu):
        k = self.kernel_size
        imu = imu.unfold(-1, k, k).permute(0, 2, 1, 3).reshape(imu.size(0), -1, 6 * k)
        tokens = self.imu_stem.norm_layer(self.imu_stem.proj(imu))
        tokens = torch.cat([self.cls_token.expand(tokens.size(0), -1, -1), tokens], 1)
        return {"trunk": {"tokens": tokens + self.pos_embed}, "head": {}}


class ImageBindIMUClassifier(nn.Module):
    """IMU branch of imagebind_huge (18.7M) + linear classifier.

    UniMTS windows are 10 s at 20 Hz; ImageBind expects 10 s at 200 Hz, 6 channels.
    Active joints are mean-pooled, gyroscope channels are zeros (acc-only protocol).
    """

    def __init__(self, ncls, ckpt=IMAGEBIND_CKPT, seq_len=2000):
        super().__init__()
        self.seq_len = seq_len
        dim, heads, blocks, out = 512, 8, 6, 1024
        self.modality_preprocessors = nn.ModuleDict({"imu": _IMUPreprocessor(dim)})
        attn = partial(_MHA, embed_dim=dim, num_heads=heads, bias=True, add_bias_kv=True)
        self.modality_trunks = nn.ModuleDict({"imu": _SimpleTransformer(
            attn, dim, blocks,
            pre_transformer_layer=nn.Sequential(nn.Identity(), _Transpose01()),
            post_transformer_layer=_Transpose01(),
        )})
        self.modality_heads = nn.ModuleDict({"imu": nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            _SelectCLS(),
            nn.Dropout(0.5),
            nn.Linear(dim, out, bias=False),
        )})
        self.head = nn.Linear(out, ncls)
        self.n_loaded, _ = _load_matching(self, _imagebind_state(ckpt))

    def encode(self, x):
        imu = to_imu6(x, self.seq_len)
        tokens = self.modality_preprocessors["imu"](imu)["trunk"]["tokens"]
        tokens = self.modality_trunks["imu"](tokens)
        return self.modality_heads["imu"](tokens)

    def forward(self, x):
        return self.head(self.encode(x))


class BioBankSSLClassifier(nn.Module):
    """OxWearables harnet10 (UK Biobank wrist-acc SSL) + EvaClassifier head.

    Input is resampled to 10 s at 30 Hz and converted to g, matching the hub model.
    """

    def __init__(self, ncls, ckpt=BIOBANK_CKPT, seq_len=300):
        super().__init__()
        self.seq_len = seq_len
        acc_path = os.path.join(BIOBANK_DIR, "sslearning", "models", "accNet.py")
        if not os.path.isfile(acc_path):
            raise FileNotFoundError("clone OxWearables/ssl-wearables into %s" % BIOBANK_DIR)
        net = _load_file("sslearning_accnet", acc_path)
        self.net = net.Resnet(output_size=int(ncls), is_eva=True, epoch_len=10, n_channels=3)
        self.n_loaded = 0
        if os.path.isfile(ckpt) and os.path.getsize(ckpt) > 10_000:
            raw = torch.load(ckpt, map_location="cpu")
            if isinstance(raw, dict) and "state_dict" in raw:
                raw = raw["state_dict"]
            renamed = {}
            for k, v in raw.items():
                nk = k[7:] if k.startswith("module.") else k
                if nk.split(".")[0] == "classifier":
                    continue
                renamed[nk] = v
            self.n_loaded, _ = _load_matching(self.net, renamed)
        self.head = self.net.classifier

    def encode(self, x):
        acc = to_wrist3(x, self.seq_len).transpose(1, 2)
        return self.net.feature_extractor(acc).flatten(1)

    def forward(self, x):
        return self.net(to_wrist3(x, self.seq_len).transpose(1, 2))


class _PosEmbed(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class _TokenEmbed(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                   padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.transpose(1, 2)).transpose(1, 2)


class GPT4TSClassifier(nn.Module):
    """Official GPT4TS classification recipe: patch embed + frozen GPT-2 (6 layers).

    Attention/FFN stay frozen; layer-norm and positional embeddings are trained,
    plus the patch embedding and a linear head (Zhou et al., NeurIPS 2023).
    """

    def __init__(self, ncls, seq_len=200, patch_size=16, stride=8, gpt_layers=6, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.pad = nn.ReplicationPad1d((0, stride))
        self.patch_num = (seq_len - patch_size) // stride + 2
        self.value_embedding = _TokenEmbed(66 * patch_size, d_model)
        self.position_embedding = _PosEmbed(d_model)
        self.emb_drop = nn.Dropout(0.1)
        gpt = _gpt_trunk(gpt_layers)
        for name, p in gpt.named_parameters():
            p.requires_grad = ("ln" in name) or ("wpe" in name)
        self.gpt2 = gpt
        self.act = nn.GELU()
        self.ln_proj = nn.LayerNorm(d_model * self.patch_num)
        self.head = nn.Linear(d_model * self.patch_num, ncls)
        self.n_loaded = gpt_layers

    def encode(self, x):
        b, _, m = x.shape
        z = self.pad(x.transpose(1, 2))
        z = z.unfold(-1, self.patch_size, self.stride)
        z = z.permute(0, 2, 3, 1).reshape(b, z.size(2), self.patch_size * m)
        z = self.emb_drop(self.value_embedding(z) + self.position_embedding(z))
        z = self.gpt2(inputs_embeds=z).last_hidden_state
        return self.act(z).reshape(b, -1)

    def forward(self, x):
        return self.head(self.ln_proj(self.encode(x)))


def build(name, ncls, **kwargs):
    name = name.lower().replace("-", "_")
    if name in ("imagebind", "imagebind_imu"):
        return ImageBindIMUClassifier(ncls, **kwargs)
    if name in ("biobank", "biobankssl", "harnet10"):
        return BioBankSSLClassifier(ncls, **kwargs)
    if name in ("gpt4ts", "gpt2ts"):
        return GPT4TSClassifier(ncls, **kwargs)
    raise ValueError("unknown sota backbone %r" % name)
