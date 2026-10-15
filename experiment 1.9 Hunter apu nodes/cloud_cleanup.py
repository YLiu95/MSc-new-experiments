import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi


SOURCES = (
    ("experiment 1.7.1/publication/relay/publication.json", "YL95/experiment-1.7.1-hunter-apu-nodes", "model"),
    ("experiment 1.7.1/staging/warmstart/download.json", "YL95/experiment-1.7-asc-gpu-nodes", "model"),
    ("experiment 1.7.1/staging/dataset/download.json", "YL95/new_experiment_1-data", "dataset"),
    ("experiment 1.9/publication/publication.json", "YL95/experiment-1.9-hunter-apu-nodes", "model"),
)


def safe_path(workspace, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe cleanup path")
    path = workspace
    for component in name.parts:
        path = path / component
        if path.is_symlink() or path.is_junction():
            raise ValueError("Cleanup refuses links and junctions")
    path.resolve().relative_to(workspace.resolve())
    return path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_candidate(workspace, relative, metadata, remote):
    path = safe_path(workspace, relative)
    expected = metadata.get("sha256", metadata.get("digest"))
    if not remote.lfs or not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
        raise ValueError("Cleanup requires a recorded SHA-256 and a cloud LFS object")
    if remote.size != metadata["bytes"] or remote.lfs.sha256 != expected:
        raise ValueError("Cloud copy does not match the recorded file")
    if path.stat().st_size != remote.size or sha256(path) != expected:
        raise ValueError("Local file does not match its verified cloud copy")
    return {"path": relative, "bytes": remote.size, "sha256": expected}


def remove_verified(workspace, entry):
    path = safe_path(workspace, entry["path"])
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError("File changed after verification; refusing deletion")
    path.unlink()


def receipt_files(receipt, remote):
    if isinstance(receipt["files"], dict):
        return receipt["files"]
    if isinstance(receipt["files"], int) and receipt["files"] > 0 and receipt.get("verified"):
        return {name: {"bytes": entry.size, "sha256": entry.lfs.sha256}
                for name, entry in remote.items() if entry.lfs and name.startswith(("best/", "latest/"))}
    raise ValueError("Unsupported cleanup receipt")


def cleanup(workspace, api, audit, delete=False):
    workspace = workspace.resolve()
    plan = []
    for relative, repository, repo_type in SOURCES:
        receipt_path = safe_path(workspace, relative)
        receipt = json.loads(receipt_path.read_text())
        if receipt["repository"] != repository:
            raise ValueError("Unexpected repository in cleanup receipt")
        if receipt_path.name == "publication.json" and (not receipt.get("verified") or receipt.get("metrics_only")):
            raise ValueError("Publication is not fully verified")
        revision = receipt.get("revision", receipt.get("commit"))
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Cleanup requires an immutable cloud revision")
        info = api.repo_info(repository, repo_type=repo_type, revision=revision, files_metadata=True)
        remote = {entry.rfilename: entry for entry in info.siblings}
        for name, metadata in receipt_files(receipt, remote).items():
            candidate = (receipt_path.parent.relative_to(workspace) / name).as_posix()
            path = safe_path(workspace, candidate)
            if metadata["bytes"] < 100 * 1024**2 or path.suffix not in {".safetensors", ".parquet"} or not path.exists():
                continue
            print(json.dumps({"event": "verify_cloud_and_local", "path": candidate, "bytes": metadata["bytes"]}), flush=True)
            entry = verify_candidate(workspace, candidate, metadata, remote[name])
            plan.append({**entry, "repository": repository, "revision": revision, "cloud_path": name})
    if not plan:
        raise ValueError("No matching large files; nothing deleted")
    before = shutil.disk_usage(workspace).free
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("x", encoding="utf-8") as stream:
        def record(event):
            event["time"] = datetime.now(timezone.utc).isoformat()
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(event), flush=True)

        record({"event": "verified_plan", "delete": delete, "files": len(plan), "bytes": sum(entry["bytes"] for entry in plan)})
        for entry in plan:
            record({"event": "verified", **entry})
            if delete:
                remove_verified(workspace, entry)
                record({"event": "deleted", **entry})
        record({"event": "complete", "delete": delete, "free_before": before,
                "free_after": shutil.disk_usage(workspace).free})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--delete", action="store_true")
    arguments = parser.parse_args()
    load_dotenv(arguments.credentials)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for cloud verification")
    cleanup(arguments.workspace, HfApi(token=token), arguments.audit, arguments.delete)