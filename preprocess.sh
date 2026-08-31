#!/usr/bin/env bash
# NeuroFL pipeline: raw BIDS -> 3D volumes for the CNN.
# Contract: $NEUROFL_SOURCE_PATH in, $NEUROFL_TARGET_PATH out (already created).
set -euo pipefail
exec python3 /site/pipelines/prepare_raw.py \
    "$NEUROFL_SOURCE_PATH" "$NEUROFL_TARGET_PATH" --shape "${NEUROFL_SHAPE:-64,64,64}"
