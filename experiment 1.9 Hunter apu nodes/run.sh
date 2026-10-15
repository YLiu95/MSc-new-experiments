#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${PBS_JOBID:?Must run inside the existing PBS allocation}"
[[ "$PBS_JOBID" == "286464.hunter-pbs01" ]] || exit 1
[[ "$(hostname -s)" == "x1000c0s1b0n0" ]] || exit 1
export HUNTER_WORKSPACE=/lustre/hpe/ws12/ws12.a/ws/vdbg1007-hunter-exp171
source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
export NCCL_NET=Socket
export PYTHONPATH="$(cd -- "$(dirname -- "$0")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
root="$HUNTER_WORKSPACE/experiment19/artifacts/job-$PBS_JOBID"
mkdir -p "$root/control"
mkdir "$root/control/launch.lock"
exec >> "$root/job.log" 2>&1
cd "$(dirname -- "$0")"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN
python -u slice_model.py --parent "$HUNTER_WORKSPACE/artifacts/job-286067.hunter-pbs01/exports/step_00000305_000000048640" --output "$HUNTER_WORKSPACE/experiment19/initial"
python -m pytest tests/test_experiment19.py -q
end_at="$(date -d '2026-10-15 16:11:06 +0200' +%s)"
exec python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 --no-python /opt/pbs/bin/pbs_attach -j "$PBS_JOBID" -s "$HUNTER_WORKSPACE/venv/bin/python" -u train19.py --root "$root" --panel "$HUNTER_WORKSPACE/data/panel" --initial "$HUNTER_WORKSPACE/experiment19/initial" --end-at "$end_at"