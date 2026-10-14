import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ranker.data import write_json
from .recovery import backup, backup_root, digest, verify


def quota_available(path):
    free = shutil.disk_usage(path).free
    if str(path).startswith("/lustre/"):
        output = subprocess.check_output(["lfs", "quota", "-g", str(os.getgid()), str(path)], text=True)
        rows = [line.split() for line in output.splitlines()]
        usage = next((row for row in rows if len(row) >= 8 and row[0].rstrip("*").isdigit()
                      and row[1].isdigit() and row[2].isdigit()), None)
        if usage is None:
            raise ValueError("Cannot verify Lustre project quota")
        used, limit = int(usage[0].rstrip("*")), int(usage[2])
        if limit:
            free = min(free, (limit - used) * 1024)
    return free


def check_space(root, parameters):
    required = parameters * 16
    if quota_available(root) < required:
        raise RuntimeError(f"Insufficient quota for an atomic recovery checkpoint: need {required} bytes free")


def export_checkpoint(root, checkpoint, name):
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    config = json.loads((checkpoint / "config.json").read_text())
    names = [f"weights-{rank:02d}.safetensors" for rank in range(config["tp"])] + ["config.json", "state.json"]
    destination = root / "exports" / checkpoint.name
    if not destination.exists():
        stage = destination.with_name(destination.name + ".staging")
        stage.mkdir(parents=True, exist_ok=False, mode=0o700)
        for filename in names:
            if filename not in manifest:
                raise ValueError(f"Missing inference export: {filename}")
            os.link(checkpoint / filename, stage / filename)
        write_json(stage / "manifest.json", {filename: manifest[filename] for filename in names})
        (stage / "COMPLETE").write_text(digest(stage / "manifest.json") + "\n")
        stage.rename(destination)
    write_json(root / "exports" / f"{name}.json", {"directory": destination.name})


def finish_migration(root, latest, copied):
    config = json.loads((latest / "config.json").read_text())
    migration = config.get("migration")
    receipt = root / "reports" / "migration_retired.json"
    if not migration or receipt.exists():
        return
    parent = Path(migration["parent_checkpoint"]).resolve()
    parent.relative_to((root.parent / "job-285983.hunter-pbs01" / "checkpoints").resolve())
    state = json.loads((latest / "state.json").read_text())
    if (state["world"] != 24 or state["step"] < migration["parent_step"]
            or state["scheduled"] < migration["parent_scheduled"]
            or digest(latest / "manifest.json") != digest(copied / "manifest.json")):
        raise ValueError("Verified migrated recovery is required before retiring the parent")
    if not parent.exists():
        write_json(receipt, {"parent_already_absent": True, "replacement": latest.name})
        return
    verify(parent, ("config.json", "state.json", "sampler.sqlite"))
    if digest(parent / "manifest.json") != migration["parent_manifest_sha256"]:
        raise ValueError("Parent checkpoint changed before migration retirement")
    parent_root = parent.parents[1]
    finished = json.loads((parent_root / "TRAINING_DONE.json").read_text())
    if finished["step"] != migration["parent_step"] or finished["scheduled"] != migration["parent_scheduled"]:
        raise ValueError("Parent training must finish before migration retirement")
    best_pointer = parent_root / "exports" / "best.json"
    if best_pointer.exists() and not (root / "best.json").exists():
        best = parent_root / "exports" / json.loads(best_pointer.read_text())["directory"]
        best.resolve().relative_to((parent_root / "exports").resolve())
        verify(best, workers=4)
        destination = root / "exports" / (best.name + "-parent")
        if not destination.exists():
            shutil.copytree(best, destination, copy_function=os.link)
        verify(destination, workers=4)
        write_json(root / "exports" / "best.json", {"directory": destination.name})
    evidence = parent_root / "reports" / "migration-source"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "state.json", "manifest.json", "COMPLETE"):
        shutil.copy2(parent / name, evidence / name)
    shutil.rmtree(parent)
    if (parent_root / "exports").exists():
        shutil.rmtree(parent_root / "exports")
    report = {"parent_manifest_sha256": migration["parent_manifest_sha256"],
              "replacement_job": root.name, "replacement": latest.name, "verified": True,
              "optimizer_preserved": True, "sampler_preserved": True, "rng_reseeded": True}
    write_json(parent_root / "reports" / "migrated_to.json", report)
    write_json(receipt, report)
    os.utime(root, None)
    print(json.dumps({"event": "migration_parent_retired", **report}), flush=True)


def maintain(root):
    started = time.monotonic()
    latest = root / json.loads((root / "latest.json").read_text())["directory"]
    latest.resolve().relative_to((root / "checkpoints").resolve())
    verify(latest)
    export_checkpoint(root, latest, "latest")
    best_pointer = root / "best.json"
    if best_pointer.exists():
        best = root / json.loads(best_pointer.read_text())["directory"]
        if best.exists():
            export_checkpoint(root, best, "best")
    for checkpoint in (root / "checkpoints").glob("step_*"):
        if checkpoint.is_dir() and checkpoint != latest and not checkpoint.name.endswith(".staging"):
            shutil.rmtree(checkpoint)
    destination = backup_root() / root.name
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    copied = destination / latest.name
    backup(latest, copied)
    for checkpoint in destination.glob("step_*"):
        if checkpoint.is_dir() and checkpoint != copied and not checkpoint.name.endswith(".staging"):
            shutil.rmtree(checkpoint)
    protected = {json.loads(pointer.read_text())["directory"] for pointer in (root / "exports").glob("*.json")}
    for export in (root / "exports").glob("step_*"):
        if export.is_dir() and export.name not in protected and not export.name.endswith(".staging"):
            shutil.rmtree(export)
    elapsed = time.monotonic() - started
    write_json(root / "reports" / "private_backup.json", {
        "verified": True, "checkpoint": latest.name, "manifest_sha256": digest(copied / "manifest.json"),
        "independent_failure_domain": False, "seconds": elapsed, "verified_epoch": time.time()})
    finish_migration(root, latest, copied)
    print(json.dumps({"event": "offline_backup_complete", "checkpoint": latest.name, "seconds": elapsed}), flush=True)
    return elapsed


def gate_complete(root):
    report = json.loads((root / "reports" / "preflight.json").read_text())
    if not report["passed"]:
        raise ValueError("Cannot complete a failed preflight")
    if report["world"] == 8:
        write_json(root.parent / "control" / "two-node-preflight.json", report)
    for path in (root / "preflight", backup_root() / root.name / "preflight"):
        if path.is_dir():
            shutil.rmtree(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("maintain", "gate-complete"))
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    {"maintain": maintain, "gate-complete": gate_complete}[arguments.action](arguments.root)