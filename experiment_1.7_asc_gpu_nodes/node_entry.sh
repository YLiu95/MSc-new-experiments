#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
master_host="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
master_address="$(getent ahostsv4 "$master_host" | awk 'NR == 1 { print $1 }')"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset GITHUB_TOKEN GH_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec python -m torch.distributed.run \
    --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=8 --node_rank="$SLURM_PROCID" \
    --master_addr="$master_address" --master_port="$((20000 + SLURM_JOB_ID % 20000))" \
    --max_restarts=0 --log-dir="$ARTIFACT_ROOT/logs/$1/node-$SLURM_PROCID" --redirects=3 --tee=0:3 \
    -m ranking17.train --root "$ARTIFACT_ROOT" \
    --panel /net/tscratch/people/tutorial042/experiments/experiment_1.6_masked/panel \
    --backup "$HOME/experiment_1.7_backups/job-$SLURM_JOB_ID" "${@:2}"