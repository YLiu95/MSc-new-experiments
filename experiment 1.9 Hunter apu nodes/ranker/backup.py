from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

from .checkpoints import digest, pointer, verify
from .data import write_json


def selected_files(root: Path) -> dict[str, Path]:
    paths = {}
    for label in ("best", "latest"):
        directory = pointer(root, label)
        for path in directory.iterdir():
            if path.is_file():
                paths[f"{label}/{path.name}"] = path
    for filename in ("config.json", "history.jsonl", "validation.jsonl", "restarts.jsonl", "TRAINING_DONE.json"):
        path = root / filename
        if path.is_file():
            paths[filename] = path
    for folder in ("reports", "runs"):
        directory = root / folder
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file():
                    paths[path.relative_to(root).as_posix()] = path
    return paths


def local(root: Path, job_id: str, all_arms: bool = False,
          backup_by: str | None = None) -> dict:
    if all_arms:
        return {arm: local(root / arm, job_id, backup_by=backup_by)
                for arm in ("A", "B", "C") if (root / arm / "best.json").exists()}
    root = root.resolve()
    backup = Path.home() / "experiment_1.6_backups" / f"job-{job_id}" / root.name
    backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup.parent.parent.chmod(0o700)
    if (backup / "COMPLETE").is_file():
        existing = json.loads((backup / "manifest.json").read_text())
        for name, data in existing["files"].items():
            path = backup / name
            if not path.is_file() or path.stat().st_size != data["bytes"] or digest(path) != data["sha256"]:
                raise ValueError("Previously completed local backup failed digest verification")
        if backup_by and datetime.fromisoformat(existing["completed_at"]).timestamp() > datetime.fromisoformat(backup_by).timestamp():
            raise RuntimeError("Previously verified backup completed after the 20-minute cutoff")
        return existing
    if backup.exists():
        raise ValueError("An interrupted local backup is preserved for inspection; choose a new destination")
    paths = selected_files(root)
    backup.mkdir(mode=0o700)
    records = {}
    for name, path in paths.items():
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        record = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if destination.stat().st_size != record["bytes"] or digest(destination) != record["sha256"]:
            raise ValueError(f"Local backup verification failed for {name}")
        records[name] = record
    manifest = {"job_id": job_id, "arm": root.name, "files": records,
                "completed_at": datetime.now().astimezone().isoformat(),
                "status": "verified", "backup_by": backup_by}
    write_json(backup / "manifest.json", manifest)
    (backup / "COMPLETE").write_text("verified\n")
    if backup_by and time.time() > datetime.fromisoformat(backup_by).timestamp():
        raise RuntimeError("Verified local backup completed after the required 20-minute cutoff")
    print(json.dumps({"event": "local_backup_verified", "arm": root.name, "job": job_id,
                      "files": len(records), "bytes": sum(entry["bytes"] for entry in records.values()),
                      "completed_at": manifest["completed_at"]}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("checkpoint", "local"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--all-arms", action="store_true")
    parser.add_argument("--backup-by")
    arguments = parser.parse_args()
    if arguments.action == "checkpoint":
        print(pointer(arguments.root, "latest"))
    elif arguments.job_id is None:
        parser.error("A local backup requires the Slurm job ID")
    else:
        local(arguments.root, arguments.job_id, arguments.all_arms, arguments.backup_by)


if __name__ == "__main__":
    main()