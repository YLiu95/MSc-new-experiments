#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
export ARTIFACT_ROOT="$ARTIFACT_ROOT/job-$SLURM_JOB_ID"
export SLURM_EXPORT_ENV=ALL
mkdir -p "$ARTIFACT_ROOT/control"
stop_file="$ARTIFACT_ROOT/control/STOP"
trap 'touch "$stop_file"' USR1 TERM
python -m ranking17.launch freeze --root "$ARTIFACT_ROOT"
stop_at="$(python -m ranking17.launch deadline --root "$ARTIFACT_ROOT")"
python -m ranking17.publish source --root "$ARTIFACT_ROOT"
srun --export=ALL --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    /usr/bin/bash ./node_entry.sh preflight --gate
python -m ranking17.publish source --root "$ARTIFACT_ROOT"
srun --export=ALL --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    /usr/bin/bash ./node_entry.sh training --stop-at "$stop_at" &
training_pid=$!
set +e
while kill -0 "$training_pid" 2>/dev/null; do
    wait "$training_pid"
    training_status=$?
done
set -e
python -m ranking17.publish backup --root "$ARTIFACT_ROOT"
python -m ranking17.publish model --root "$ARTIFACT_ROOT"
python -m ranking17.publish source --root "$ARTIFACT_ROOT"
exit "${training_status:-1}"