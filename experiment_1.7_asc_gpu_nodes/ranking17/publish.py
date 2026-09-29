import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, CommitOperationAdd
import requests

from ranker.data import write_json
from ranker.publish import credential, git_blob, git_blob_file
from .recovery import backup, digest, verify


GITHUB = "YLiu95/MSc-new-experiments"
FOLDER = "experiment_1.7_asc_gpu_nodes"
HF = "YL95/experiment-1.7-asc-gpu-nodes"
SOURCE = Path(__file__).resolve().parents[1]


def source(root):
    hub = HfApi(token=credential("HF_TOKEN"))
    if hub.whoami()["name"] != "YL95":
        raise ValueError("Wrong HF account")
    hub.create_repo(HF, repo_type="model", private=False, exist_ok=True)
    if hub.model_info(HF).private:
        raise ValueError("Expected public inference destination")
    files = {}
    for folder in ("ranking17", "ranker", "ascgpu", "tests"):
        for path in (SOURCE / folder).glob("*.py"):
            files[f"{FOLDER}/{folder}/{path.name}"] = path.read_bytes()
    for name in ("README.md", "MODEL_CARD.md", "WORK_PROCESS_REPORT.md", "env.sh", "node_entry.sh",
                 "batch_entry.sh", "requirements.txt", "requirements.lock.txt"):
        files[f"{FOLDER}/{name}"] = (SOURCE / name).read_bytes()
    files[f"{FOLDER}/EXPERIMENT_PLAN.md"] = (Path.home() / "Experiment 1.7 plan.md").read_bytes()
    files[f"{FOLDER}/configs/{root.name}.json"] = (root / "config.json").read_bytes()
    for name in ("preflight.json", "private_backup.json", "public_backup.json"):
        path = root / "reports" / name
        if path.is_file():
            files[f"{FOLDER}/reports/{root.name}-{name}"] = path.read_bytes()
    if (root / "TRAINING_DONE.json").is_file():
        files[f"{FOLDER}/reports/{root.name}-training.json"] = (root / "TRAINING_DONE.json").read_bytes()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {credential('GITHUB_TOKEN')}", "Accept": "application/vnd.github+json"})

    def api(method, path, **kwargs):
        response = session.request(method, f"https://api.github.com/repos/{GITHUB}{path}", timeout=45, **kwargs)
        if not response.ok:
            raise RuntimeError(f"GitHub request failed: HTTP {response.status_code}")
        return response.json()

    repository = api("GET", "")
    if not repository.get("permissions", {}).get("push"):
        raise ValueError("Source repository is not writable")
    branch = repository["default_branch"]
    head = api("GET", f"/git/ref/heads/{branch}")["object"]["sha"]
    previous_tree = api("GET", f"/git/commits/{head}")["tree"]["sha"]
    entries = []
    for name, content in files.items():
        blob = api("POST", "/git/blobs", json={"content": base64.b64encode(content).decode(), "encoding": "base64"})["sha"]
        if blob != git_blob(content):
            raise ValueError("Source digest mismatch")
        entries.append({"path": name, "mode": "100644", "type": "blob", "sha": blob})
    tree = api("POST", "/git/trees", json={"base_tree": previous_tree, "tree": entries})["sha"]
    commit = api("POST", "/git/commits", json={"message": "Experiment 1.7 source and aggregate verification", "tree": tree, "parents": [head]})["sha"]
    api("PATCH", f"/git/refs/heads/{branch}", json={"sha": commit, "force": False})
    verified_tree = api("GET", f"/git/trees/{tree}?recursive=1")
    if verified_tree.get("truncated"):
        raise ValueError("Truncated source verification")
    remote = {entry["path"]: entry["sha"] for entry in verified_tree["tree"]}
    if any(remote.get(name) != git_blob(content) for name, content in files.items()):
        raise ValueError("Source verification failed")
    if api("GET", f"/git/ref/heads/{branch}")["object"]["sha"] != commit:
        raise ValueError("Branch moved during source verification")
    visibility = api("GET", "")["private"]
    if visibility != repository["private"]:
        raise ValueError("Repository visibility changed")
    report = {"repository": GITHUB, "private": visibility, "branch": branch, "commit": commit,
              "files": {name: git_blob(content) for name, content in files.items()},
              "verified_at": datetime.now().astimezone().isoformat()}
    write_json(root / "control" / "github.json", report)
    print(json.dumps({key: report[key] for key in ("repository", "private", "branch", "commit")}), flush=True)


def pointer(root, name):
    location = (root / json.loads((root / f"{name}.json").read_text())["directory"]).resolve()
    location.relative_to((root / "checkpoints").resolve())
    verify(location)
    return location


def private_backup(root):
    destination = Path.home() / "experiment_1.7_backups" / root.name
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    selected = {}
    for name in ("best", "latest"):
        location = pointer(root, name)
        copied = destination / location.name
        backup(location, copied)
        selected[name] = {"checkpoint": location.name, "manifest_sha256": digest(copied / "manifest.json")}
    if (root / "runs").is_dir():
        shutil.copytree(root / "runs", destination / "runs", dirs_exist_ok=True)
        for path in (root / "runs").glob("events.out.tfevents.*"):
            if digest(path) != digest(destination / "runs" / path.name):
                raise ValueError("Private aggregate event backup mismatch")
    report = {"checkpoints": selected, "verified": True, "verified_at": datetime.now().astimezone().isoformat()}
    write_json(root / "reports" / "private_backup.json", report)
    print(json.dumps(report), flush=True)


def model(root):
    if not (root / "TRAINING_DONE.json").is_file():
        raise ValueError("Public export requires orderly training completion")
    if not json.loads((root / "reports" / "private_backup.json").read_text())["verified"]:
        raise ValueError("Public export requires a verified private backup")
    best, latest = pointer(root, "best"), pointer(root, "latest")
    if json.loads((best / "state.json").read_text())["step"] <= 0:
        raise ValueError("No trained best checkpoint")
    paths = {f"best/{path.name}": path for path in best.glob("weights-*.safetensors")}
    paths.update({f"best/{name}": best / name for name in ("config.json", "state.json")})
    publication = root / "publication"
    publication.mkdir(exist_ok=True, mode=0o700)
    if best != latest:
        from safetensors.torch import save_file
        import torch
        for shard in range(8):
            recovery = torch.load(latest / f"recovery-{shard:02d}.pt", map_location="cpu", weights_only=True)
            weights = {key: value.to(dtype=torch.bfloat16 if value.ndim >= 2 else torch.float32).contiguous()
                       for key, value in recovery["model"].items()}
            path = publication / f"latest-weights-{shard:02d}.safetensors"
            save_file(weights, str(path))
            paths[f"latest/weights-{shard:02d}.safetensors"] = path
            del recovery, weights
        paths.update({f"latest/{name}": latest / name for name in ("config.json", "state.json")})
    paths["README.md"] = SOURCE / "MODEL_CARD.md"
    for path in (root / "runs").glob("events.out.tfevents.*"):
        paths[f"runs/{path.name}"] = path
    for name in ("preflight.json", "private_backup.json"):
        paths[f"reports/{name}"] = root / "reports" / name
    paths["reports/training.json"] = root / "TRAINING_DONE.json"
    for name in ("history.jsonl", "validation.jsonl"):
        if (root / name).is_file():
            paths[f"reports/{name}"] = root / name
    api = HfApi(token=credential("HF_TOKEN"))
    if api.whoami()["name"] != "YL95":
        raise ValueError("Wrong HF account")
    api.create_repo(HF, repo_type="model", private=False, exist_ok=True)
    if api.model_info(HF).private:
        raise ValueError("Expected public inference destination")
    commit = api.create_commit(HF, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                                              for name, path in paths.items()],
                               commit_message="Verified Experiment 1.7 selected inference exports and aggregates", num_threads=8)
    remote = {entry.rfilename: entry for entry in api.model_info(HF, revision=commit.oid, files_metadata=True).siblings}
    verified = {}
    for name, path in paths.items():
        entry = remote[name]
        expected = digest(path) if entry.lfs is not None else git_blob_file(path)
        actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        if entry.size != path.stat().st_size or actual != expected:
            raise ValueError("HF backup size/digest mismatch")
        verified[name] = {"bytes": entry.size, "digest": actual}
    write_json(root / "reports" / "public_backup.json", {"repository": HF, "commit": commit.oid,
               "private": False, "files": verified, "best_equals_latest": best == latest,
               "verified_at": datetime.now().astimezone().isoformat()})
    print(json.dumps({"repository": HF, "commit": commit.oid, "files": len(verified)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("source", "backup", "model"))
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    {"source": source, "backup": private_backup, "model": model}[arguments.action](arguments.root)