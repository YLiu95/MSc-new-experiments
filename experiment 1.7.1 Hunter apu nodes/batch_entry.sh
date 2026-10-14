#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "${HUNTER_SOURCE:-$HUNTER_WORKSPACE/source}"
source ./env.sh
export ARTIFACT_ROOT="$HUNTER_WORKSPACE/artifacts/job-$PBS_JOBID"
mkdir -p "$ARTIFACT_ROOT/control"
exec >> "$ARTIFACT_ROOT/job.log" 2>&1
export HUNTER_NODES="$(sort -u "$PBS_NODEFILE" | wc -l)"
export MASTER_ADDR="$(head -n 1 "$PBS_NODEFILE")"
export MASTER_PORT="$((20000 + ${PBS_JOBID%%.*} % 20000))"
trap 'touch "$ARTIFACT_ROOT/control/STOP"' TERM INT
if [[ "$HUNTER_MODE" == "preflight" && ! -f "$HUNTER_WORKSPACE/data/panel/meta.json" ]]; then
    python -m ranker.data --root "$HUNTER_WORKSPACE/data" --snapshot "$HUNTER_WORKSPACE/dataset"
fi
if [[ "$HUNTER_MODE" == "preflight" ]]; then
    python -m ranker.evaluate register --root "$HUNTER_WORKSPACE/data"
fi
migration_args=()
if [[ -n "${HUNTER_MIGRATE_FROM:-}" ]]; then
    migration_args=(--migrate-from "$HUNTER_MIGRATE_FROM")
fi
python -m ranking17.launch freeze --root "$ARTIFACT_ROOT" --nodes "$HUNTER_NODES" "${migration_args[@]}"
stop_at="$(python -m ranking17.launch deadline --root "$ARTIFACT_ROOT")"
mpiexec -n "$HUNTER_NODES" --ppn 1 --cpu-bind none --gpu-bind none \
    /usr/bin/bash ./node_entry.sh preflight --gate --stop-at "$stop_at"
python -m ranking17.offline gate-complete --root "$ARTIFACT_ROOT"
if [[ "$HUNTER_MODE" == "preflight" ]]; then
    exit 0
fi
mpiexec -n "$HUNTER_NODES" --ppn 1 --cpu-bind none --gpu-bind none \
    /usr/bin/bash ./node_entry.sh training --stop-at "$stop_at" "${migration_args[@]}"
python -m ranking17.offline maintain --root "$ARTIFACT_ROOT"