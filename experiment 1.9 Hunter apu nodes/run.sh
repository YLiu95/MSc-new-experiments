#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${PBS_JOBID:?Must run inside the existing PBS allocation}"
if [[ "$PBS_JOBID" == "286615.hunter-pbs01" ]]; then
	[[ "$(hostname -s)" == "x1000c2s1b0n0" ]] || exit 1
	export HUNTER_WORKSPACE=/lustre/hpe/ws12/ws12.a/ws/vdbg1007-hunter-exp171
	source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
	export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TZ=Europe/Berlin
	export TRAIN19_SOURCE="$(cd -- "$(dirname -- "$0")" && pwd)"
	export PYTHONPATH="$TRAIN19_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
	export ARTIFACT_ROOT="$HUNTER_WORKSPACE/experiment19/artifacts/job-$PBS_JOBID"
	export MASTER_ADDR="$(head -n 1 "$PBS_NODEFILE")"
	export MASTER_PORT="$((20000 + ${PBS_JOBID%%.*} % 20000))"
	export TRAIN_END_AT="$(date -d "$(qstat -f -w "$PBS_JOBID" | sed -n 's/^[[:space:]]*stime = //p') + 5 hours" +%s)"
	export RESUME_CHECKPOINT="$HUNTER_WORKSPACE/experiment19/artifacts/job-286464.hunter-pbs01/checkpoints/step_00000066_000000021120"
	mkdir -p "$ARTIFACT_ROOT/control"
	unset RESTART_CHECKPOINT
	if [[ "${1:-}" == "--restart" ]]; then
		[[ "$#" == 2 && -f "$2/COMPLETE" ]] || exit 1
		export RESTART_CHECKPOINT="$2"
		mkdir "$ARTIFACT_ROOT/control/restart-validation.lock"
	else
		[[ "$#" == 0 ]] || exit 1
		mkdir "$ARTIFACT_ROOT/control/launch.lock"
	fi
	unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN
	mpiexec -n 2 --ppn 1 --cpu-bind none --gpu-bind none /usr/bin/bash -c '
		set -euo pipefail
		source "$HUNTER_WORKSPACE/source-six-node-20261014/env.sh"
		export NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1
		restore_args=(--resume "$RESUME_CHECKPOINT")
		log_directory="$ARTIFACT_ROOT/logs/node-$PMI_RANK"
		if [[ -n "${RESTART_CHECKPOINT:-}" ]]; then
			restore_args=(--restart "$RESTART_CHECKPOINT")
			log_directory="$ARTIFACT_ROOT/logs/restart-validation/node-$PMI_RANK"
		fi
		exec "$HUNTER_WORKSPACE/venv/bin/python" -m torch.distributed.run \
			--nnodes=2 --nproc_per_node=4 --node_rank="$PMI_RANK" \
			--master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
			--max_restarts=0 --log-dir="$log_directory" --redirects=3 --tee=0:3 \
			--no-python /opt/pbs/bin/pbs_attach -j "$PBS_JOBID" -s "$HUNTER_WORKSPACE/venv/bin/python" -u \
			"$TRAIN19_SOURCE/train19.py" --root "$ARTIFACT_ROOT" --panel "$HUNTER_WORKSPACE/data/panel" \
			--initial "$HUNTER_WORKSPACE/experiment19/initial" --end-at "$TRAIN_END_AT" \
			"${restore_args[@]}"
	' 2>&1 | while IFS= read -r line; do
		printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$line"
	done | tee -a "$ARTIFACT_ROOT/job.log"
	exit "${PIPESTATUS[0]}"
fi
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