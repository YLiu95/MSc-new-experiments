#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "${PBS_JOBID:-}" == "286972.hunter-pbs01" ]] || exit 1
: "${HUNTER_WORKSPACE:?}" "${PBS_NODEFILE:?}" "${TRAIN_END_AT:?}"
source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
export TRAIN191_SOURCE="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$TRAIN191_SOURCE:$HUNTER_WORKSPACE/source-six-node-20261014${PYTHONPATH:+:$PYTHONPATH}"
export ARTIFACT_ROOT="$HUNTER_WORKSPACE/experiment191/artifacts/job-$PBS_JOBID"
export MASTER_ADDR="$(head -n 1 "$PBS_NODEFILE")"
export MASTER_PORT="$((20000 + ${PBS_JOBID%%.*} % 20000))"
export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TZ=Europe/Berlin
export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
[[ "$(sort -u "$PBS_NODEFILE" | wc -l)" == 12 ]] || exit 1
[[ "$(hostname -s)" == "${MASTER_ADDR%%.*}" ]] || exit 1
mkdir -p "$ARTIFACT_ROOT/control"
mkdir "$ARTIFACT_ROOT/control/run.lock"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN
mpiexec -n 12 --ppn 1 --cpu-bind none --gpu-bind none /usr/bin/bash -c '
    set -euo pipefail
    source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
    export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
    exec "$HUNTER_WORKSPACE/venv/bin/python" -m torch.distributed.run \
        --nnodes=12 --nproc_per_node=4 --node_rank="$PMI_RANK" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --max_restarts=0 \
        --log-dir="$ARTIFACT_ROOT/logs/node-$PMI_RANK" --redirects=3 --tee=0:3 \
        --no-python /opt/pbs/bin/pbs_attach -j "$PBS_JOBID" -s \
        "$HUNTER_WORKSPACE/venv/bin/python" -u "$TRAIN191_SOURCE/train191.py" \
        --root "$ARTIFACT_ROOT" --panel "$HUNTER_WORKSPACE/data/panel" \
        --parent "$HUNTER_WORKSPACE/experiment19/artifacts/job-286615.hunter-pbs01/checkpoints/step_00000328_000000104960" \
        --end-at "$TRAIN_END_AT"
' 2>&1 | while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$line"
done | tee -a "$ARTIFACT_ROOT/job.log"
exit "${PIPESTATUS[0]}"