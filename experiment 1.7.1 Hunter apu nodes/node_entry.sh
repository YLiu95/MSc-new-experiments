#!/usr/bin/env bash
set -euo pipefail
cd "$HUNTER_WORKSPACE/source"
source ./env.sh
node_rank="${PMI_RANK:?MPI launcher rank is required}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset GITHUB_TOKEN GH_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN HTTP_PROXY HTTPS_PROXY ALL_PROXY
exec python -m torch.distributed.run \
    --nnodes="$HUNTER_NODES" --nproc_per_node=4 --node_rank="$node_rank" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --max_restarts=0 --log-dir="$ARTIFACT_ROOT/logs/$1/node-$node_rank" --redirects=3 --tee=0:3 \
    -m ranking17.train --root "$ARTIFACT_ROOT" \
    --panel "$HUNTER_WORKSPACE/data/panel" \
    --warm-start "$HUNTER_WORKSPACE/warmstart" \
    --backup "$BACKUP_ROOT/job-$PBS_JOBID" "${@:2}"