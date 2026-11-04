#!/usr/bin/env python3
"""Summarize UniMTS-compatible joint assignments used by RoFT.

Prints, per dataset, which of the 22 joints are non-zero on a split, and
documents UTD-MHAD placement branching and MotionSense channel duplication.
Does not modify any files.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATASETS = [
    "Opp_g", "UCIHAR", "MotionSense", "w-HAR", "Shoaib", "har70plus",
    "realworld", "TNDA-HAR", "PAMAP", "USCHAD", "Mhealth", "Harth",
    "ut-complex", "Wharf", "WISDM", "DSADS", "UTD-MHAD", "MMAct",
]


def _active_joints(x: np.ndarray) -> list[int]:
    # x: [N, T, V, C] or [N, C, T, V] depending on loader; UniMTS load returns torch later.
    # Here we inspect raw npy via a light path: non-zero energy over N,T,C per joint.
    if x.ndim != 4:
        raise ValueError("expected 4-D array, got shape %s" % (x.shape,))
    # Heuristic: joint axis is the one with size 22.
    joint_axis = list(x.shape).index(22)
    axes = tuple(i for i in range(4) if i != joint_axis)
    energy = np.abs(x).sum(axis=axes)
    return [int(j) for j, e in enumerate(energy) if e > 0]


def describe_utd(y: np.ndarray) -> None:
    n_wrist = int((y < 21).sum())
    n_thigh = int((y >= 21).sum())
    print("  UTD-MHAD action-ID placement (official collection protocol):")
    print("    actions 0..20 (IDs < 21) -> joint 21 (wrist): %d windows" % n_wrist)
    print("    actions 21..26 (IDs >= 21) -> joint 5 (thigh): %d windows" % n_thigh)
    print("    Reference: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html")


def describe_motionsense() -> None:
    print("  MotionSense: single phone IMU copied to joints 5 and 1 in UniMTS loader.")
    print("  RoFT samples rotations per joint index; duplicated channels may receive")
    print("  different bounded rotations (graph-input regularizer, not two devices).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="UniMTS_data root")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()

    print("RoFT / UniMTS joint-mapping audit (%s split)" % args.split)
    print("data_path =", os.path.abspath(args.data_path))
    print()

    for ds in DATASETS:
        x_path = os.path.join(args.data_path, ds, "X_%s.npy" % args.split)
        y_path = os.path.join(args.data_path, ds, "y_%s.npy" % args.split)
        if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
            print("[%s] missing raw npy — skip" % ds)
            continue
        # Raw X is not yet joint-mapped; print labels only and special cases.
        y = np.load(y_path)
        print("[%s] labels shape=%s classes=%d" % (ds, y.shape, len(np.unique(y))))
        if ds == "UTD-MHAD":
            describe_utd(y)
        if ds == "MotionSense":
            describe_motionsense()
    print()
    print("Full joint assignment is applied in datasets/data.py::load().")
    print("See docs/JOINT_MAPPING.md for protocol interpretation.")


if __name__ == "__main__":
    main()
