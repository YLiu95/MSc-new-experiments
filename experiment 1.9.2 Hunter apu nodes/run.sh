#!/usr/bin/env bash
#PBS -N exp192_6n_24h
#PBS -q route
#PBS -l select=6:node_type=mi300a:ncpus=96:mpiprocs=1
#PBS -l walltime=24:00:00
#PBS -l place=scatter:excl
#PBS -r n
#PBS -j oe
#PBS -m n
#PBS -S /bin/bash
set -euo pipefail
umask 077
[[ "${PBS_JOBID:-}" =~ ^[0-9]+\.hunter-pbs01$ ]] || exit 1
[[ "${PBS_ENVIRONMENT:-}" == PBS_BATCH || ( "${PBS_ENVIRONMENT:-}" == PBS_INTERACTIVE && "${TRAIN192_ATTACHED:-}" == "$PBS_JOBID" ) ]] || exit 1
: "${HUNTER_WORKSPACE:?}" "${PBS_NODEFILE:?}" "${TRAIN192_SOURCE:?}"
export EXPERIMENT192_JOB="$PBS_JOBID"
source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
export PYTHONPATH="$TRAIN192_SOURCE:$HUNTER_WORKSPACE/source-six-node-20261014${PYTHONPATH:+:$PYTHONPATH}"
export EXPERIMENT192_ROOT="$HUNTER_WORKSPACE/experiment192/artifacts/job-$PBS_JOBID"
export PYTHONUNBUFFERED=1
cd "$TRAIN192_SOURCE"
record_exit() {
    result=$?
    trap - EXIT
    "$HUNTER_WORKSPACE/venv/bin/python" "$TRAIN192_SOURCE/control192.py" \
        --workspace "$HUNTER_WORKSPACE" --source "$TRAIN192_SOURCE" --batch-exit "$result" || true
    exit "$result"
}
trap record_exit EXIT
TRAIN_END_AT="$("$HUNTER_WORKSPACE/venv/bin/python" "$TRAIN192_SOURCE/control192.py" \
    --workspace "$HUNTER_WORKSPACE" --source "$TRAIN192_SOURCE" --batch-start)"
export TRAIN_END_AT
export PREPARED192="$HUNTER_WORKSPACE/experiment192/prepared"
export PARENT192="$HUNTER_WORKSPACE/experiment19/artifacts/job-286615.hunter-pbs01/checkpoints/step_00000328_000000104960"
mapfile -t hosts < <(awk '!seen[$0]++' "$PBS_NODEFILE")
[[ "${#hosts[@]}" == 6 ]] || exit 1
[[ "$(hostname -s)" == "${hosts[0]%%.*}" ]] || exit 1
export MASTER_A="${hosts[0]}" MASTER_B="${hosts[3]}"
export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TZ=Europe/Berlin
export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
[[ -f "$PREPARED192/COMPLETE" ]] || exit 1
mkdir -p "$EXPERIMENT192_ROOT/control" "$EXPERIMENT192_ROOT/arm-A/control" "$EXPERIMENT192_ROOT/arm-B/control"
mkdir "$EXPERIMENT192_ROOT/control/run.lock"
mkdir "$EXPERIMENT192_ROOT/arm-A/control/run.lock" "$EXPERIMENT192_ROOT/arm-B/control/run.lock"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN
mpiexec -n 6 --ppn 1 --cpu-bind none --gpu-bind none /usr/bin/bash -c '
    set -euo pipefail
    source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
    if (( PMI_RANK < 3 )); then
        export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
        arm=A; master="$MASTER_A"; port=29192
    else
        export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
        arm=B; master="$MASTER_B"; port=29193
    fi
    node_rank=$((PMI_RANK % 3))
    root="$EXPERIMENT192_ROOT/arm-$arm"
    if (( PMI_RANK == 5 )); then
        "$HUNTER_WORKSPACE/venv/bin/python" "$TRAIN192_SOURCE/ops192.py" audits-local \
            --workspace "$HUNTER_WORKSPACE" --source "$TRAIN192_SOURCE"
    fi
    resume_arguments=()
    if [[ -n "${RESUME192_ROOT:-}" ]]; then
        resume_arguments=(--resume-arm-root "$RESUME192_ROOT/arm-$arm")
    fi
    exec "$HUNTER_WORKSPACE/venv/bin/python" -m torch.distributed.run \
        --nnodes=3 --nproc_per_node=4 --node_rank="$node_rank" \
        --master_addr="$master" --master_port="$port" --max_restarts=0 \
        --log-dir="$root/logs/node-$node_rank" --redirects=3 --tee=0:3 \
        "$TRAIN192_SOURCE/train192.py" \
        --root "$root" --panel "$HUNTER_WORKSPACE/data/panel" --parent "$PARENT192" \
        --prepared "$PREPARED192" --arm "$arm" --end-at "$TRAIN_END_AT" "${resume_arguments[@]}"
' 2>&1 | while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$line"
done | tee -a "$EXPERIMENT192_ROOT/job.log"
if ! "$HUNTER_WORKSPACE/venv/bin/python" -c 'import os, sys; from pathlib import Path; from evaluate192 import final_ready; sys.exit(0 if final_ready(Path(os.environ["EXPERIMENT192_ROOT"])) else 1)'; then
    printf '%s\n' 'Training paused with recovery saved; final test remains sealed.' | tee -a "$EXPERIMENT192_ROOT/job.log"
    exit 0
fi
mpiexec -n 6 --ppn 1 --cpu-bind none --gpu-bind none /usr/bin/bash -c '
    set -euo pipefail
    source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
    export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
    exec "$HUNTER_WORKSPACE/venv/bin/python" -m torch.distributed.run \
        --nnodes=6 --nproc_per_node=4 --node_rank="$PMI_RANK" \
        --master_addr="$MASTER_A" --master_port=29202 --max_restarts=0 \
        --log-dir="$EXPERIMENT192_ROOT/final-logs/node-$PMI_RANK" --redirects=3 --tee=0:3 \
        "$TRAIN192_SOURCE/final192.py" \
        --root "$EXPERIMENT192_ROOT" --panel "$HUNTER_WORKSPACE/data/panel" \
        --parent "$PARENT192" --prepared "$PREPARED192" --end-at "$TRAIN_END_AT" --final-evaluation
' 2>&1 | tee -a "$EXPERIMENT192_ROOT/final-evaluation.log"