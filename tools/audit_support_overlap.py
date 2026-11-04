#!/usr/bin/env python3
"""Audit official UniMTS few-shot support indices vs train/test arrays.

Reports, per dataset and k:
  - number of support indices
  - whether all indices lie in the train split range
  - whether support indices are nested across increasing k (when files exist)

Does not modify any files.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

DATASETS = [
    "Opp_g", "UCIHAR", "MotionSense", "w-HAR", "Shoaib", "har70plus",
    "realworld", "TNDA-HAR", "PAMAP", "USCHAD", "Mhealth", "Harth",
    "ut-complex", "Wharf", "WISDM", "DSADS", "UTD-MHAD", "MMAct",
]
KS = [1, 2, 3, 5, 10]


def load_indices(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("indices", "idx", "index", "support_idx"):
            if key in obj:
                return np.asarray(obj[key]).reshape(-1)
        # common UniMTS pattern: dict of class -> index list
        vals = []
        for v in obj.values():
            vals.extend(list(np.asarray(v).reshape(-1)))
        return np.asarray(vals, dtype=np.int64)
    return np.asarray(obj).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--few_shot_dir", default="few_shot_data_2")
    args = ap.parse_args()

    fs_root = os.path.join(args.data_path, args.few_shot_dir)
    print("Support-index audit")
    print("data_path =", os.path.abspath(args.data_path))
    print("few_shot_dir =", fs_root)
    print()

    for ds in DATASETS:
        y_train = os.path.join(args.data_path, ds, "y_train.npy")
        if not os.path.isfile(y_train):
            print("[%s] missing y_train.npy — skip" % ds)
            continue
        n_train = len(np.load(y_train))
        prev = None
        print("[%s] n_train=%d" % (ds, n_train))
        for k in KS:
            # UniMTS filenames vary slightly; try both patterns.
            candidates = [
                os.path.join(fs_root, "%s_k=%d.pth" % (ds, k)),
                os.path.join(fs_root, "%s_k%d.pth" % (ds, k)),
            ]
            path = next((p for p in candidates if os.path.isfile(p)), None)
            if path is None:
                print("  k=%d: missing support file" % k)
                continue
            idx = load_indices(path)
            in_range = bool(idx.min() >= 0 and idx.max() < n_train) if len(idx) else False
            nested = "n/a"
            if prev is not None:
                nested = str(set(prev.tolist()).issubset(set(idx.tolist())))
            print(
                "  k=%d: n_idx=%d in_train_range=%s nested_from_smaller_k=%s file=%s"
                % (k, len(idx), in_range, nested, os.path.basename(path))
            )
            prev = idx
        print()


if __name__ == "__main__":
    main()
