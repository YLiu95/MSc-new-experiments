import argparse
import base64
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
import requests

from ranker.data import write_json
from ranker.publish import git_blob, git_blob_file
from ranking17 import publish as legacy
from ranking17.recovery import digest


SOURCE = Path(__file__).resolve().parent
GITHUB = "YLiu95/MSc-new-experiments"
FOLDER = "experiment 1.9 Hunter apu nodes"
HF = "YL95/experiment-1.9-hunter-apu-nodes"
BASE_SNAPSHOT = legacy.public_snapshot


def public_snapshot(workspace):
    snapshot = BASE_SNAPSHOT(workspace)
    if not snapshot.get("job"):
        return snapshot
    root = workspace / "artifacts" / snapshot["job"]
    additions = {
        "source/runtime-source.tar.gz": workspace / "provenance" / "runtime-source.tar.gz",
        "source/deployed-source.tar.gz": workspace / "source-fixed.tar.gz",
        f"reports/{root.name}/config.json": root / "config.json",
        f"reports/{root.name}/checkpoint_verification.json": root / "reports" / "checkpoint_verification.json",
        f"reports/{root.name}/scheduler.json": root / "reports" / "scheduler.json",
    }
    for name, path in additions.items():
        if path.is_file():
            snapshot["files"][name] = {"path": path.relative_to(workspace).as_posix(),
                                       "bytes": path.stat().st_size, "sha256": digest(path)}
    return snapshot


def runtime_files(archive):
    files = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            allowed = (name.parts[0] in {"ranker", "ascgpu", "ranking17"} and name.suffix == ".py") or str(name) in {
                "env.sh", "requirements.hunter.txt"}
            if name.is_absolute() or ".." in name.parts or not member.isfile() or not allowed:
                raise ValueError("Unexpected member in runtime-source archive")
            files[name.as_posix()] = source.extractfile(member).read()
    if not {"ranking17/model.py", "ranker/train.py", "ascgpu/model.py"} <= files.keys():
        raise ValueError("Runtime-source archive is incomplete")
    return files


def snapshot_at(endpoint):
    response = requests.get(endpoint + "/snapshot", timeout=60)
    response.raise_for_status()
    snapshot = response.json()
    if snapshot.get("job") != "job-286464.hunter-pbs01":
        raise ValueError("Unexpected experiment job at publication endpoint")
    return snapshot


def publish_source(root, endpoint):
    snapshot = snapshot_at(endpoint)
    archive = legacy.cache_public_file(root, "source/runtime-source.tar.gz",
                                      snapshot["files"]["source/runtime-source.tar.gz"], endpoint)
    files = runtime_files(archive)
    for name in ("slice_model.py", "train19.py", "run.sh", "publish19.py", "README.md", "MODEL_CARD.md",
                 "tests/test_experiment19.py"):
        files[name] = (SOURCE / name).read_bytes()
    files["provenance/runtime-source.sha256"] = (digest(archive) + "  runtime-source.tar.gz\n").encode()
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + legacy.credential("GITHUB_TOKEN"),
                            "Accept": "application/vnd.github+json"})

    def api(method, path, **kwargs):
        response = session.request(method, f"https://api.github.com/repos/{GITHUB}{path}", timeout=60, **kwargs)
        if not response.ok:
            raise RuntimeError(f"GitHub request failed: HTTP {response.status_code}")
        return response.json()

    if not api("GET", "").get("permissions", {}).get("push"):
        raise ValueError("GitHub repository is not writable")
    head = api("GET", "/git/ref/heads/main")["object"]["sha"]
    previous_tree = api("GET", f"/git/commits/{head}")["tree"]["sha"]
    entries = []
    for name, content in files.items():
        blob = api("POST", "/git/blobs", json={"content": base64.b64encode(content).decode(), "encoding": "base64"})["sha"]
        if blob != git_blob(content):
            raise ValueError("GitHub blob verification failed")
        entries.append({"path": f"{FOLDER}/{name}", "mode": "100755" if name.endswith(".sh") else "100644",
                        "type": "blob", "sha": blob})
    tree = api("POST", "/git/trees", json={"base_tree": previous_tree, "tree": entries})["sha"]
    commit = api("POST", "/git/commits", json={"message": "Back up Experiment 1.9 depth-pruned Hunter ranker",
                                              "tree": tree, "parents": [head]})["sha"]
    api("PATCH", "/git/refs/heads/main", json={"sha": commit, "force": False})
    remote_tree = api("GET", f"/git/trees/{tree}?recursive=1")
    remote = {entry["path"]: entry["sha"] for entry in remote_tree["tree"]}
    if remote_tree.get("truncated") or any(remote.get(f"{FOLDER}/{name}") != git_blob(content)
                                           for name, content in files.items()):
        raise ValueError("GitHub tree verification failed")
    report = {"repository": GITHUB, "folder": FOLDER, "branch": "main", "commit": commit,
              "files": len(files), "verified": True, "time": datetime.now().astimezone().isoformat()}
    write_json(root / "github.json", report)
    print(json.dumps(report), flush=True)


def publish_model(root, endpoint, metrics_only=False):
    api = HfApi(token=legacy.credential("HF_TOKEN"))
    if api.whoami()["name"] != "YL95":
        raise ValueError("Wrong Hugging Face account")
    api.create_repo(HF, repo_type="model", private=False, exist_ok=True)
    if api.model_info(HF).private:
        raise ValueError("Expected public inference repository")
    snapshot = snapshot_at(endpoint)
    paths = {"README.md": SOURCE / "MODEL_CARD.md"}
    cached = {}
    for name, entry in snapshot["files"].items():
        if metrics_only and name.startswith(("best/", "latest/", "source/")):
            continue
        print(json.dumps({"event": "download", "file": name, "bytes": entry["bytes"]}), flush=True)
        if entry["sha256"] in cached:
            paths[name] = cached[entry["sha256"]]
        else:
            paths[name] = legacy.cache_public_file(root, name, entry, endpoint)
            cached[entry["sha256"]] = paths[name]
    if not metrics_only and not {"best/weights-00.safetensors", "latest/weights-00.safetensors"} <= paths.keys():
        raise ValueError("Both best and latest inference checkpoints are required")
    commit = api.create_commit(HF, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                                              for name, path in paths.items()],
                               commit_message="Experiment 1.9 TensorBoard and metrics" if metrics_only else
                               "Verified Experiment 1.9 best/latest inference checkpoints and metrics", num_threads=4)
    remote = {entry.rfilename: entry for entry in api.model_info(HF, revision=commit.oid, files_metadata=True).siblings}
    verified = {}
    for name, path in paths.items():
        entry = remote[name]
        actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        expected = digest(path) if entry.lfs is not None else git_blob_file(path)
        if entry.size != path.stat().st_size or actual != expected:
            raise ValueError(f"Remote file verification failed: {name}")
        verified[name] = {"bytes": entry.size, "digest": actual}
    report = {"repository": HF, "commit": commit.oid, "job": snapshot["job"], "verified": True,
              "metrics_only": metrics_only, "files": verified, "time": datetime.now().astimezone().isoformat()}
    write_json(root / ("metrics.json" if metrics_only else "publication.json"), report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "source", "model"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18767")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--metrics-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "serve":
        legacy.public_snapshot = public_snapshot
        legacy.serve(arguments.root, arguments.socket)
    else:
        if arguments.credentials:
            load_dotenv(arguments.credentials)
        if arguments.action == "source":
            publish_source(arguments.root, arguments.endpoint)
        else:
            publish_model(arguments.root, arguments.endpoint, arguments.metrics_only)