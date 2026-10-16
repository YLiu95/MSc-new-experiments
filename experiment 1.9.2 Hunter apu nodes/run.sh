#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "${PBS_JOBID:-}" == "287925.hunter-pbs01" ]] || exit 1
: "${HUNTER_WORKSPACE:?}" "${PBS_NODEFILE:?}" "${TRAIN_END_AT:?}"
source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
export TRAIN192_SOURCE="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$TRAIN192_SOURCE:$HUNTER_WORKSPACE/source-six-node-20261014${PYTHONPATH:+:$PYTHONPATH}"
export EXPERIMENT192_ROOT="$HUNTER_WORKSPACE/experiment192/artifacts/job-$PBS_JOBID"
export PREPARED192="$HUNTER_WORKSPACE/experiment192/prepared"
export PARENT192="$HUNTER_WORKSPACE/experiment19/artifacts/job-286615.hunter-pbs01/checkpoints/step_00000328_000000104960"
mapfile -t hosts < <(awk '!seen[$0]++' "$PBS_NODEFILE")
[[ "${#hosts[@]}" == 12 ]] || exit 1
[[ "$(hostname -s)" == "${hosts[0]%%.*}" ]] || exit 1
export MASTER_A="${hosts[0]}" MASTER_B="${hosts[6]}"
export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TZ=Europe/Berlin
export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
[[ -f "$PREPARED192/COMPLETE" ]] || exit 1
mkdir -p "$EXPERIMENT192_ROOT/control" "$EXPERIMENT192_ROOT/arm-A/control" "$EXPERIMENT192_ROOT/arm-B/control"
mkdir "$EXPERIMENT192_ROOT/control/run.lock"
mkdir "$EXPERIMENT192_ROOT/arm-A/control/run.lock" "$EXPERIMENT192_ROOT/arm-B/control/run.lock"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN
mpiexec -n 12 --ppn 1 --cpu-bind none --gpu-bind none /usr/bin/bash -c '
    set -euo pipefail
    source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
    if (( PMI_RANK < 6 )); then
        export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
        arm=A; master="$MASTER_A"; port=29192
    else
        export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
        arm=B; master="$MASTER_B"; port=29193
    fi
    node_rank=$((PMI_RANK % 6))
    root="$EXPERIMENT192_ROOT/arm-$arm"
    exec "$HUNTER_WORKSPACE/venv/bin/python" -m torch.distributed.run \
        --nnodes=6 --nproc_per_node=4 --node_rank="$node_rank" \
        --master_addr="$master" --master_port="$port" --max_restarts=0 \
        --log-dir="$root/logs/node-$node_rank" --redirects=3 --tee=0:3 \
        --no-python /opt/pbs/bin/pbs_attach -j "$PBS_JOBID" -s \
        "$HUNTER_WORKSPACE/venv/bin/python" -u "$TRAIN192_SOURCE/train192.py" \
        --root "$root" --panel "$HUNTER_WORKSPACE/data/panel" --parent "$PARENT192" \
        --prepared "$PREPARED192" --arm "$arm" --end-at "$TRAIN_END_AT"
' 2>&1 | while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$line"
done | tee -a "$EXPERIMENT192_ROOT/job.log"
mpiexec -n 12 --ppn 1 --cpu-bind none --gpu-bind none /usr/bin/bash -c '
    set -euo pipefail
    source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
    export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
    exec "$HUNTER_WORKSPACE/venv/bin/python" -m torch.distributed.run \
        --nnodes=12 --nproc_per_node=4 --node_rank="$PMI_RANK" \
        --master_addr="$MASTER_A" --master_port=29202 --max_restarts=0 \
        --log-dir="$EXPERIMENT192_ROOT/final-logs/node-$PMI_RANK" --redirects=3 --tee=0:3 \
        --no-python /opt/pbs/bin/pbs_attach -j "$PBS_JOBID" -s \
        "$HUNTER_WORKSPACE/venv/bin/python" -u "$TRAIN192_SOURCE/final192.py" \
        --root "$EXPERIMENT192_ROOT" --panel "$HUNTER_WORKSPACE/data/panel" \
        --parent "$PARENT192" --prepared "$PREPARED192" --end-at "$TRAIN_END_AT" --final-evaluation
' 2>&1 | tee -a "$EXPERIMENT192_ROOT/final-evaluation.log"