#!/usr/bin/env bash
# Run protocol audit helpers once UniMTS_data is available.
set -euo pipefail
DATA_PATH="${1:-./UniMTS_data}"
echo "Using data_path=${DATA_PATH}"
python tools/audit_joint_mapping.py --data_path "${DATA_PATH}" --split test
python tools/audit_support_overlap.py --data_path "${DATA_PATH}"
echo "Done. See docs/PROTOCOL.md for the full evaluation specification."
