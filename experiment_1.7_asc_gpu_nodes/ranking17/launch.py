import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from torch._subclasses.fake_tensor import FakeTensorMode

from ranker.checkpoints import digest
from ranker.data import write_json
from .model import ModelConfig, RankingModel, ParallelContext
from .recovery import backup_root


BASE = Path("/net/tscratch/people/tutorial042/experiments/experiment_1.7")
PANEL = Path("/net/tscratch/people/tutorial042/experiments/experiment_1.6_masked/panel")
SOURCE = Path(__file__).resolve().parents[1]


def fields(line):
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def reservations():
    output = subprocess.check_output(["scontrol", "show", "reservation", "-o"], text=True)
    return [fields(line) for line in output.splitlines() if line.strip()]


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def freeze(root, nodes):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    metadata = json.loads((PANEL / "meta.json").read_text())
    model_config = ModelConfig(tickers=metadata["n_tickers"])
    with FakeTensorMode():
        model = RankingModel(model_config, ParallelContext(tp_size=8))
        parameters = model.parameter_count()
    source_hash = hashlib.sha256()
    for folder in ("ranking17", "ranker", "ascgpu"):
        for path in sorted((SOURCE / folder).glob("*.py")):
            source_hash.update(path.relative_to(SOURCE).as_posix().encode())
            source_hash.update(path.read_bytes())
    config = {"experiment": "1.7", "model": asdict(model_config), "parameters": parameters,
              "tp": 8, "dp": nodes, "global_batch": 320, "microbatch": 1,
              "data_meta_sha256": digest(PANEL / "meta.json"),
              "vocabulary_sha256": digest(PANEL / "vocabulary.json"),
              "validation_sha256": digest(PANEL / "evaluation" / "val-primary.jsonl"),
              "environment_lock_sha256": digest(SOURCE / "requirements.lock.txt"),
              "source_sha256": source_hash.hexdigest(), "warm_start": False,
              "task_configurations": 638820, "reserved_training_tasks": [], "nonrepetition": True,
              "validation": "frozen 1.6 primary monitoring subset; no full-validation claim",
              "master_weights": "float32", "matrix_compute": "bfloat16",
              "private_backup_root": str(backup_root()), "backup_independent_failure_domain": False,
              "shutdown_reserve_seconds": 1200}
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Refusing to change an already frozen configuration")
    write_json(path, config)
    return config


def deadline(root):
    job = fields(subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True))
    reservation = next(item for item in reservations() if item["ReservationName"] == job["Reservation"])
    end = min(epoch(job["EndTime"]), epoch(reservation["EndTime"]))
    stop = end - 1200
    if stop - time.time() < 900:
        raise ValueError("Insufficient time for preflight and protected shutdown")
    write_json(root / "control" / "deadlines.json", {"job_end": job["EndTime"],
               "reservation_end": reservation["EndTime"], "stop_epoch": stop, "backup_by_epoch": end,
               "job_id": os.environ["SLURM_JOB_ID"]})
    print(stop, flush=True)


def submit():
    BASE.mkdir(parents=True, exist_ok=True, mode=0o700)
    BASE.chmod(0o700)
    control = BASE / "control"
    control.mkdir(exist_ok=True)
    active = [item for item in reservations() if item.get("State") == "ACTIVE"
              and "tutorial" in item.get("Accounts", "").split(",")]
    if not active:
        raise ValueError("No active tutorial reservation; no submission made")
    selected = max(active, key=lambda item: epoch(item["EndTime"]))
    minutes = min(600, math.floor((epoch(selected["EndTime"]) - time.time() - 120) / 60))
    if minutes < 45:
        raise ValueError("Too little reservation time remains")
    environment = dict(os.environ)
    for name in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE", "SALLOC_USE_MIN_NODES"):
        environment.pop(name, None)
    command = ["sbatch", "--parsable", "--export=NONE", f"--deadline={selected['EndTime']}", "--partition=tutorial",
               "--account=tutorial", f"--reservation={selected['ReservationName']}", "--nodes=2-8",
               "--ntasks-per-node=1", "--cpus-per-task=32", "--gpus-per-node=8", "--mem=256G",
               f"--time={minutes}", "--signal=B:USR1@1200", "--job-name=experiment-1.7",
               f"--chdir={SOURCE}", f"--output={control}/job-%j.out", f"--error={control}/job-%j.err",
               str(SOURCE / "batch_entry.sh")]
    current = next(item for item in reservations() if item["ReservationName"] == selected["ReservationName"])
    if current["State"] != "ACTIVE" or current["EndTime"] != selected["EndTime"]:
        raise ValueError("Reservation changed before submission")
    output = subprocess.check_output(command, env=environment, text=True).strip()
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise ValueError("No valid submitted job ID")
    write_json(control / "submission.json", {"job_id": job_id, "reservation": current,
               "minutes": minutes, "command": command, "submitted_at": datetime.now().astimezone().isoformat()})
    print(json.dumps({"job_id": job_id, "minutes": minutes, "root": str(BASE / f"job-{job_id}")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "freeze", "deadline"))
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "submit":
        submit()
    elif arguments.action == "freeze":
        print(json.dumps(freeze(arguments.root, int(os.environ["SLURM_JOB_NUM_NODES"]))))
    else:
        deadline(arguments.root)