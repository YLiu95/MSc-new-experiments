#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ./env.sh
python -c 'import ctypes, json, sqlite3, ssl, numpy, pyarrow, torch; print(json.dumps({"python_ssl": ssl.OPENSSL_VERSION, "sqlite": sqlite3.sqlite_version, "torch": torch.__version__, "cuda_wheel": torch.version.cuda, "numpy": numpy.__version__, "pyarrow": pyarrow.__version__, "status": "runtime_loaded"}))'