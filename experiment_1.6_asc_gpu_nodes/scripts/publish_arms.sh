#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
python -m scripts.publish_arms --root "$ARTIFACT_ROOT" --screen-job-id "$1"