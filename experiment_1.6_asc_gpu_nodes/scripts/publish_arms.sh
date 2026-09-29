#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
umask 077
export HF_HOME="$ARTIFACT_ROOT/cache/hf-publish"
mkdir -p "$HF_HOME"
python -m scripts.publish_arms --root "$ARTIFACT_ROOT" --screen-job-id "$1"