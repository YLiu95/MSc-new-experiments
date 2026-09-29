#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
export SLURM_EXPORT_ENV=ALL
if [[ "${1:-}" == "--groups" ]]; then
    if [[ "$SLURM_JOB_NUM_NODES" -ne 8 ]]; then
        exit 2
    fi
    srun --cpu-bind=none --cpus-per-task=32 --nodes="$SLURM_JOB_NUM_NODES" \
        --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
        /usr/bin/bash "$SLURM_SUBMIT_DIR/node_entry.sh" '-' '-' \
        "$ARTIFACT_ROOT/control/diagnostic-stop-$SLURM_JOB_ID" --diagnostic
else
    srun --cpu-bind=none --nodes=1 --ntasks=1 --cpus-per-task=2 \
        /usr/bin/bash "$SLURM_SUBMIT_DIR/scripts/runtime_preflight.sh"
fi