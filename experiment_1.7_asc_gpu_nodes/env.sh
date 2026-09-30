#!/usr/bin/env bash
set -euo pipefail
python_env="${EXPERIMENT_PYTHON_ENV:-/net/tscratch/people/$(id -un)/experiments/experiment_1.7_venv}"
if [[ ! -x "$python_env/bin/python" ]]; then
    printf 'Missing Experiment 1.7 Python environment\n' >&2
    exit 1
fi
export PATH="$python_env/bin:$PATH"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/net/tscratch/people/$(id -un)/experiments/experiment_1.7}"
export BACKUP_ROOT="${BACKUP_ROOT:-/net/tscratch/people/$(id -un)/experiments/experiment_1.7_backups}"
export PYTHONPATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
IFS=: read -r -a library_paths <<< "${LD_LIBRARY_PATH:-}"
filtered_library_path=""
for runtime_path in \
    /net/software/v1/software/Python/3.11.5-GCCcore-13.2.0/lib \
    /net/software/v1/software/OpenSSL/1.1/lib \
    /net/software/v1/software/SQLite/3.43.1-GCCcore-13.2.0/lib \
    /net/software/v1/software/libffi/3.4.4-GCCcore-13.2.0/lib64 \
    /net/software/v1/software/zlib/1.2.13-GCCcore-13.2.0/lib; do
    if [[ ! -d "$runtime_path" ]]; then
        printf 'Missing required runtime path: %s\n' "$runtime_path" >&2
        exit 1
    fi
    filtered_library_path="${filtered_library_path:+$filtered_library_path:}$runtime_path"
done
for library_path in "${library_paths[@]}"; do
    case "$library_path" in
        */CUDA/*|*/NCCL/*|*/NVHPC/*|*/UCX-CUDA/*|*/UCC-CUDA/*|*/GDRCopy/*) continue ;;
    esac
    if [[ -n "$library_path" ]]; then
        filtered_library_path="${filtered_library_path:+$filtered_library_path:}$library_path"
    fi
done
export LD_LIBRARY_PATH="$filtered_library_path"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1