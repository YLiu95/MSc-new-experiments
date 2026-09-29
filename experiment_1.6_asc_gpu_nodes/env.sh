#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.venvs/experiment-1.6-gpu/bin:$PATH"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/net/tscratch/people/$(id -un)/experiments/experiment_1.6_masked}"
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
export PYTHONUNBUFFERED=1