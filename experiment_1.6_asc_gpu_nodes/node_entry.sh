#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
unset HF_TOKEN GITHUB_TOKEN GH_TOKEN HUGGING_FACE_HUB_TOKEN
stop_at="$1"
backup_by="$2"
stop_file="$3"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t groups < <(python -m ranker.layout --nodes "$SLURM_JOB_NUM_NODES" --process-id "$SLURM_PROCID")
if [[ "${#hosts[@]}" -ne "$SLURM_JOB_NUM_NODES" || "${#groups[@]}" -eq 0 ]]; then
    exit 2
fi
for group in "${groups[@]}"; do
    IFS=: read -r arm group_nodes node_rank leader_index <<< "$group"
    if [[ -e "$stop_file" ]]; then
        break
    fi
    master_address="$(getent ahostsv4 "${hosts[$leader_index]}" | awk 'NR == 1 { print $1 }')"
    if [[ -z "$master_address" ]]; then
        exit 2
    fi
    case "$arm" in
        A) arm_index=0 ;;
        B) arm_index=1 ;;
        C) arm_index=2 ;;
        *) exit 2 ;;
    esac
    master_port="$((20000 + SLURM_JOB_ID % 20000 + arm_index))"
    log_directory="$ARTIFACT_ROOT/$arm/logs/torchrun-$SLURM_JOB_ID/node-$node_rank"
    mkdir -p "$log_directory"
    if [[ "${4:-}" == "--diagnostic" ]]; then
        python -m torch.distributed.run --nnodes="$group_nodes" --nproc-per-node=8 \
            --node-rank="$node_rank" --master-addr="$master_address" --master-port="$master_port" \
            --max-restarts=0 --log-dir="$log_directory" --redirects=3 --tee=0:3 \
            -m ranker.parallel_check --device cuda
        continue
    fi
    resume=()
    if [[ -f "$ARTIFACT_ROOT/$arm/latest.json" ]]; then
        checkpoint="$(python -m ranker.backup checkpoint --root "$ARTIFACT_ROOT/$arm")"
        resume=(--resume "$checkpoint")
    fi
    python -m torch.distributed.run --nnodes="$group_nodes" --nproc-per-node=8 \
        --node-rank="$node_rank" --master-addr="$master_address" --master-port="$master_port" \
        --max-restarts=0 --log-dir="$log_directory" --redirects=3 --tee=0:3 \
        -m ranker.train --panel "$ARTIFACT_ROOT/panel" --root "$ARTIFACT_ROOT" --arm "$arm" \
        --stop-at "$stop_at" --backup-by "$backup_by" --stop-file "$stop_file" "${resume[@]}"
    if [[ "$node_rank" -eq 0 ]]; then
        python -m ranker.backup local --root "$ARTIFACT_ROOT/$arm" --job-id "$SLURM_JOB_ID" --backup-by "$backup_by"
    fi
done