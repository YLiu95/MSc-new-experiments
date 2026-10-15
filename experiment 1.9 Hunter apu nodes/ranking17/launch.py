import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from torch._subclasses.fake_tensor import FakeTensorMode

from ranker.checkpoints import digest
from ranker.data import write_json
from .model import ModelConfig, RankingModel, ParallelContext
from .offline import quota_available
from .recovery import backup_root, validate_migration_config, verify
from .pbs import (PRODUCTION_END, QUEUE_WAIT_SECONDS, free_apu_nodes, production_nodes,
                  running_deadlines, validate_nodes, walltime)


WORKSPACE = Path(os.environ["HUNTER_WORKSPACE"])
BASE = WORKSPACE / "artifacts"
PANEL = WORKSPACE / "data" / "panel"
SOURCE = Path(__file__).resolve().parents[1]


def job_info(job_id):
    return json.loads(subprocess.check_output(["qstat", "-f", "-F", "json", job_id], text=True))["Jobs"][job_id]


def freeze(root, nodes, migrate_from=None):
    validate_nodes(nodes)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    metadata = json.loads((PANEL / "meta.json").read_text())
    model_config = ModelConfig(tickers=metadata["n_tickers"])
    with FakeTensorMode():
        model = RankingModel(model_config, ParallelContext(tp_size=4))
        parameters = model.parameter_count()
    source_hash = hashlib.sha256()
    for folder in ("ranking17", "ranker", "ascgpu"):
        for path in sorted((SOURCE / folder).glob("*.py")):
            source_hash.update(path.relative_to(SOURCE).as_posix().encode())
            source_hash.update(path.read_bytes())
    for name in ("env.sh", "node_entry.sh", "batch_entry.sh", "requirements.hunter.txt"):
        source_hash.update(name.encode())
        source_hash.update((SOURCE / name).read_bytes())
    parent = json.loads((WORKSPACE / "warmstart" / "latest" / "config.json").read_text())
    for name, path in (("vocabulary_sha256", PANEL / "vocabulary.json"),
                       ("validation_sha256", PANEL / "evaluation" / "val-primary.jsonl")):
        if digest(path) != parent[name]:
            raise ValueError(f"Rebuilt data differs from the original frozen contract: {name}")
    import torch
    config = {"experiment": "1.7.1", "model": asdict(model_config), "parameters": parameters,
              "tp": 4, "dp": nodes, "global_batch": 320, "microbatch": 1,
              "data_meta_sha256": digest(PANEL / "meta.json"),
              "vocabulary_sha256": digest(PANEL / "vocabulary.json"),
              "validation_sha256": digest(PANEL / "evaluation" / "val-primary.jsonl"),
              "environment_lock_sha256": digest(SOURCE / "requirements.hunter.txt"),
              "source_sha256": source_hash.hexdigest(), "warm_start": True,
              "warm_start_manifest_sha256": digest(WORKSPACE / "warmstart" / "download.json"),
              "parent_step": 153, "optimizer_reset": True, "sampler_reset": True,
              "rng_reset": True, "possible_sample_reuse": True, "exact_continuation": False,
              "torch_version": torch.__version__, "hip_version": torch.version.hip,
              "task_configurations": 638820, "reserved_training_tasks": [], "nonrepetition": True,
              "validation": "frozen 1.6 primary monitoring subset; no full-validation claim",
              "master_weights": "float32", "matrix_compute": "bfloat16",
              "private_backup_root": str(backup_root()), "backup_independent_failure_domain": False,
              "shutdown_reserve_seconds": 1200}
    if migrate_from is not None:
        migrate_from = migrate_from.resolve()
        migrate_from.relative_to((BASE / "job-285983.hunter-pbs01" / "checkpoints").resolve())
        verify(migrate_from, ("config.json", "state.json", "sampler.sqlite"))
        previous = json.loads((migrate_from / "config.json").read_text())
        state = json.loads((migrate_from / "state.json").read_text())
        finished = json.loads((migrate_from.parents[1] / "TRAINING_DONE.json").read_text())
        if any(state[key] != finished[key] for key in ("step", "scheduled", "world")):
            raise ValueError("Migration requires the final completed source checkpoint")
        config.update(warm_start=False, optimizer_reset=False, sampler_reset=False, possible_sample_reuse=False,
                      migration={"parent_checkpoint": str(migrate_from), "parent_step": state["step"],
                                 "parent_scheduled": state["scheduled"], "parent_world": state["world"],
                                 "parent_source_sha256": previous["source_sha256"],
                                 "parent_manifest_sha256": digest(migrate_from / "manifest.json"),
                                 "optimizer_preserved": True, "sampler_preserved": True,
                                 "rng_reseeded": True, "bitwise_continuation": False})
        validate_migration_config(previous, config)
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Refusing to change an already frozen configuration")
    write_json(path, config)
    return config


def deadline(root):
    job_id = os.environ["PBS_JOBID"]
    report = running_deadlines(job_info(job_id), time.time())
    write_json(root / "control" / "deadlines.json", {**report, "job_id": job_id})
    print(report["stop_epoch"], flush=True)


def submit(nodes=None, mode="preflight", migrate_from=None, waive_fraction=False):
    BASE.mkdir(parents=True, exist_ok=True, mode=0o700)
    BASE.chmod(0o700)
    control = BASE / "control"
    control.mkdir(exist_ok=True)
    if (migrate_from is not None or waive_fraction) and mode != "production":
        raise ValueError("Migration and policy waivers require production mode")
    if nodes == 6 and migrate_from is None:
        raise ValueError("Six-node production requires the approved checkpoint migration")
    if mode == "production":
        gate = json.loads((control / "two-node-preflight.json").read_text())
        if not gate["passed"] or gate["world"] != 8:
            raise ValueError("Successful two-node preflight required")
        available = free_apu_nodes(json.loads(subprocess.check_output(["pbsnodes", "-a", "-F", "json"], text=True)))
        nodes = production_nodes(available, requested=nodes, waive_fraction=waive_fraction)
        if migrate_from is not None:
            parent_id = migrate_from.resolve().parents[1].name.removeprefix("job-")
            parent_job = json.loads(subprocess.check_output(
                ["qstat", "-x", "-f", "-F", "json", parent_id], text=True))["Jobs"]
            if next(iter(parent_job.values()))["job_state"] != "F":
                raise ValueError("The source allocation must finish before submitting its migration")
        seconds = min(86400, math.floor(PRODUCTION_END - time.time() - 60))
    else:
        nodes = 2 if nodes is None else nodes
        available = None
        seconds = 3600
    validate_nodes(nodes)
    if seconds < 3600 or time.time() + seconds > PRODUCTION_END:
        raise ValueError("Insufficient time before the absolute deadline")
    if mode == "production" or (PANEL / "meta.json").is_file():
        ready = freeze(control / f"ready-{mode}-{time.time_ns()}", nodes, migrate_from)
        expected_source = ready["migration"]["parent_source_sha256"] if migrate_from else ready["source_sha256"]
        if mode == "production" and gate["source_sha256"] != expected_source:
            raise ValueError("Code changed after the two-node preflight; rerun the test")
    elif not (WORKSPACE / "dataset" / "download.json").is_file() or not (WORKSPACE / "warmstart" / "download.json").is_file():
        raise ValueError("Verified dataset and warm-start artifacts must be staged before requesting nodes")
    environment = f"HUNTER_WORKSPACE={WORKSPACE},HUNTER_SOURCE={SOURCE},HUNTER_MODE={mode}"
    if migrate_from is not None:
        environment += f",HUNTER_MIGRATE_FROM={migrate_from.resolve()}"
    command = ["qsub", "-N", f"exp171-{mode}", "-l", f"select={nodes}:node_type=mi300a:mpiprocs=1",
               "-l", f"walltime={walltime(seconds)}", "-l", "ws12=True", "-j", "oe",
               "-o", str(control / f"pbs-{mode}.log"), "-v",
               environment, str(SOURCE / "batch_entry.sh")]
    submitted = time.time()
    job_id = subprocess.check_output(command, cwd=SOURCE, text=True).strip()
    if not job_id.split(".")[0].isdigit():
        raise ValueError("Invalid PBS job identifier")
    report = {"job_id": job_id, "nodes": nodes, "free_apus": available, "seconds": seconds,
              "mode": mode, "command": command, "submitted_epoch": submitted,
              "free_node_fraction_waived": waive_fraction,
              "migration_from": str(migrate_from) if migrate_from else None,
              "queue_deadline_epoch": submitted + QUEUE_WAIT_SECONDS}
    write_json(control / f"submission-{job_id}.json", report)
    with (control / f"queue-watch-{job_id}.log").open("a") as log:
        subprocess.Popen([sys.executable, "-m", "ranking17.launch", "watch", "--job-id", job_id,
                          "--submitted", str(submitted)], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)
    print(json.dumps(report), flush=True)


def handoff(root, nodes, waive_fraction=False):
    if nodes != 6 or root.resolve() != (BASE / "job-285983.hunter-pbs01").resolve():
        raise ValueError("Handoff is restricted to the approved two-to-six-node transition")
    if not (root / "control" / "STOP").is_file():
        raise ValueError("Request a graceful source-job stop before handoff")
    control = BASE / "control"
    lock = control / "six-node-handoff.lock"
    lock.mkdir()
    wake = threading.Event()
    limit = min(time.time() + 1200, PRODUCTION_END - 3600)
    parent_id = root.name.removeprefix("job-")
    while True:
        job = next(iter(json.loads(subprocess.check_output(
            ["qstat", "-x", "-f", "-F", "json", parent_id], text=True))["Jobs"].values()))
        print(json.dumps({"event": "handoff_wait", "parent_job": parent_id, "state": job["job_state"]}), flush=True)
        if job["job_state"] == "F":
            if job.get("Exit_status") != 0:
                raise ValueError("Source job did not finish successfully")
            break
        if time.time() >= limit:
            raise ValueError("Source shutdown did not finish within the handoff budget")
        wake.wait(15)
    checkpoint = (root / json.loads((root / "latest.json").read_text())["directory"]).resolve()
    checkpoint.relative_to((root / "checkpoints").resolve())
    ready = freeze(control / f"ready-handoff-{time.time_ns()}", nodes, checkpoint)
    gate = json.loads((control / "two-node-preflight.json").read_text())
    if not gate["passed"] or gate["world"] != 8 or gate["source_sha256"] != ready["migration"]["parent_source_sha256"]:
        raise ValueError("Source checkpoint must match the verified two-node gate")
    free = free_apu_nodes(json.loads(subprocess.check_output(["pbsnodes", "-a", "-F", "json"], text=True)))
    production_nodes(free, requested=nodes, waive_fraction=waive_fraction)
    copied = backup_root() / root.name / checkpoint.name
    print(json.dumps({"event": "handoff_verify", "checkpoint": checkpoint.name}), flush=True)
    manifest = verify(checkpoint, workers=4)
    if verify(copied, workers=4) != manifest:
        raise ValueError("The final backup differs from the source checkpoint")
    payload_bytes = sum(entry["bytes"] for entry in manifest.values())
    if quota_available(root) + payload_bytes < 2 * payload_bytes + 20 * 1024**3:
        raise ValueError("Insufficient quota for the full six-node gate while retaining the parent")
    evidence = root / "reports" / "retired-redundant-backup"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "state.json", "manifest.json", "COMPLETE"):
        shutil.copy2(copied / name, evidence / name)
    receipt = {"parent_job": parent_id, "checkpoint": str(checkpoint), "verified": True,
               "manifest_sha256": digest(checkpoint / "manifest.json"), "backup_removed": str(copied),
               "freed_payload_bytes": payload_bytes, "free_node_fraction_waived": waive_fraction}
    shutil.rmtree(copied)
    write_json(control / "six-node-handoff.json", receipt)
    print(json.dumps({"event": "handoff_source_preserved", **receipt}), flush=True)
    submit(nodes, "production", checkpoint, waive_fraction)


def watch(job_id, submitted):
    wake = threading.Event()
    while True:
        try:
            state = job_info(job_id)["job_state"]
        except subprocess.CalledProcessError:
            subprocess.run(["qdel", job_id], check=False)
            return
        if state == "R":
            print(json.dumps({"job_id": job_id, "running": True}), flush=True)
            return
        if state in ("F", "E", "X"):
            return
        remaining = submitted + QUEUE_WAIT_SECONDS - time.time()
        if remaining <= 0:
            subprocess.run(["qdel", job_id], check=True)
            print(json.dumps({"job_id": job_id, "cancelled": "queue wait exceeded 600 seconds"}), flush=True)
            return
        wake.wait(min(5, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "freeze", "deadline", "watch", "handoff"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--mode", choices=("preflight", "production"), default="preflight")
    parser.add_argument("--migrate-from", type=Path)
    parser.add_argument("--waive-free-node-fraction", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--submitted", type=float)
    arguments = parser.parse_args()
    if arguments.action == "submit":
        submit(arguments.nodes, arguments.mode, arguments.migrate_from, arguments.waive_free_node_fraction)
    elif arguments.action == "handoff":
        handoff(arguments.root, arguments.nodes, arguments.waive_free_node_fraction)
    elif arguments.action == "freeze":
        print(json.dumps(freeze(arguments.root, arguments.nodes, arguments.migrate_from)))
    elif arguments.action == "watch":
        watch(arguments.job_id, arguments.submitted)
    else:
        deadline(arguments.root)