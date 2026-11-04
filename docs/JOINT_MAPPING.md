# Joint Mapping Notes (Inherited from UniMTS)

RoFT uses the UniMTS data loaders in `datasets/data.py` without changing how raw IMU windows are assigned to the 22-joint skeleton. This page documents two mappings that matter for protocol audits.

## UTD-MHAD placement

The official [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) collection protocol places the wearable IMU on the **right wrist for actions 1–21** and on the **right thigh for actions 22–27**.

The UniMTS loader encodes that placement by routing windows with `y < 21` to joint index 21 (wrist) and windows with `y >= 21` to joint index 5 (thigh):

```python
# datasets/data.py (UniMTS-compatible)
all_X[real_labels < 21, :, 21, :] = ...   # wrist
all_X[real_labels >= 21, :, 5, :]  = ...   # thigh
```

Interpretation for audits:

- The branch keys on the **dataset action ID**, which is tied to a **known collection placement** in the UTD-MHAD documentation.
- Under the UniMTS protocol we follow, this is treated as **placement metadata associated with the action vocabulary**, not as a free use of held-out semantic labels invented by RoFT.
- For deployments where placement is known a priori (watch vs. thigh tag), an equivalent mask can be supplied without reading class names. `tools/audit_joint_mapping.py` prints the mapping table and can emit a placement-only mask summary.

RoFT does **not** add any further label-conditioned preprocessing on top of this loader.

## MotionSense duplicated joints

MotionSense provides a single phone IMU. The UniMTS loader copies the same 6-D signal onto two skeleton joints (indices 5 and 1) so the graph encoder sees a multi-joint tensor consistent with other datasets.

During RoFT fine-tuning, rotations are sampled **per active joint index**. On MotionSense this means the two copied streams may receive different bounded rotations even though they originate from one physical device. We keep this behavior for protocol compatibility with UniMTS; it should be read as a graph-input regularizer on duplicated channels, not as two independent wearable mountings.

## Audit command

```bash
python tools/audit_joint_mapping.py --data_path /path/to/UniMTS_data
```

The script summarizes, per dataset, which joint indices are non-zero on the train split and flags UTD-MHAD / MotionSense specially. It does not modify any arrays.
