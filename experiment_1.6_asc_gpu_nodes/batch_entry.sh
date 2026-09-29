#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
export SLURM_EXPORT_ENV=ALL
umask 077
mkdir -p "$ARTIFACT_ROOT/control"
stop_file="$ARTIFACT_ROOT/control/stop-$SLURM_JOB_ID"
trap 'touch "$stop_file"' USR1
mapfile -t limits < <(python -m ranker.deadlines --job-id "$SLURM_JOB_ID")
if [[ "${#limits[@]}" -ne 2 ]]; then
    exit 2
fi
set +e
srun --cpu-bind=none --cpus-per-task=32 --nodes="$SLURM_JOB_NUM_NODES" \
    --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    /usr/bin/bash "$SLURM_SUBMIT_DIR/node_entry.sh" "${limits[0]}" "${limits[1]}" "$stop_file"
status=$?
set -e
python -m ranker.backup local --root "$ARTIFACT_ROOT" --job-id "$SLURM_JOB_ID" --all-arms --backup-by "${limits[1]}"
exit "$status"