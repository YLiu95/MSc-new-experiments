#!/usr/bin/env bash
set -euo pipefail
: "${HUNTER_WORKSPACE:?Set HUNTER_WORKSPACE to the allocated Lustre workspace}"
module use /opt/hlrs/stack/ai-frameworks/modulefiles
module load pytorch/2.10.0_cray-python-3.11.7_rocm-6.4.1
if [[ -x "$HUNTER_WORKSPACE/venv/bin/python" ]]; then
    source "$HUNTER_WORKSPACE/venv/bin/activate"
fi
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HUNTER_WORKSPACE/artifacts}"
export BACKUP_ROOT="${BACKUP_ROOT:-$HUNTER_WORKSPACE/backups}"
export PYTHONPATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export HF_HOME="$HUNTER_WORKSPACE/cache/hf"
export XDG_CACHE_HOME="$HUNTER_WORKSPACE/cache"
export TMPDIR="$HUNTER_WORKSPACE/tmp"
export TORCH_HOME="$HUNTER_WORKSPACE/cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ARTIFACT_ROOT" "$BACKUP_ROOT"