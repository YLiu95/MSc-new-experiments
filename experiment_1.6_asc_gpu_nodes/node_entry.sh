#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
unset HF_TOKEN GITHUB_TOKEN GH_TOKEN HUGGING_FACE_HUB_TOKEN
stop_at="$1"
backup_by="$2"
stop_file="$3"
case "$SLURM_JOB_NUM_NODES:$SLURM_PROCID" in
    1:0) arms=(A B C) ;;
    2:0) arms=(B A) ;;
    2:1) arms=(C) ;;
    3:0) arms=(A) ;;
    3:1) arms=(B) ;;
    3:2) arms=(C) ;;
    *) exit 2 ;;
esac
for arm in "${arms[@]}"; do
    if [[ -e "$stop_file" ]]; then
        break
    fi
    resume=()
    if [[ -f "$ARTIFACT_ROOT/$arm/latest.json" ]]; then
        checkpoint="$(python -m ranker.backup checkpoint --root "$ARTIFACT_ROOT/$arm")"
        resume=(--resume "$checkpoint")
    fi
    mkdir -p "$ARTIFACT_ROOT/$arm/logs/torchrun-$SLURM_JOB_ID"
    python -m torch.distributed.run --standalone --nproc-per-node=8 --max-restarts=0 \
        --log-dir="$ARTIFACT_ROOT/$arm/logs/torchrun-$SLURM_JOB_ID" --redirects=3 --tee=0:3 \
        -m ranker.train --panel "$ARTIFACT_ROOT/panel" --root "$ARTIFACT_ROOT" --arm "$arm" \
        --stop-at "$stop_at" --backup-by "$backup_by" --stop-file "$stop_file" "${resume[@]}"
    python -m ranker.backup local --root "$ARTIFACT_ROOT/$arm" --job-id "$SLURM_JOB_ID" --backup-by "$backup_by"
done