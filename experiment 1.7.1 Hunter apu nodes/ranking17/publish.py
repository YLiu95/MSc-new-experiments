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
from ranker.publish import git_blob, git_blob_file
from .recovery import backup, backup_root, digest, verify


GITHUB = "YLiu95/MSc-new-experiments"
FOLDER = "experiment 1.7.1 Hunter apu nodes"
HF = "YL95/experiment-1.7.1-hunter-apu-nodes"
SOURCE = Path(__file__).resolve().parents[1]


def credential(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing local {name} credential")
    return value


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
                 "batch_entry.sh", "requirements.txt", "requirements.hunter.txt"):
        files[f"{FOLDER}/{name}"] = (SOURCE / name).read_bytes()
    files[f"{FOLDER}/EXPERIMENT_PLAN.md"] = (SOURCE / "EXPERIMENT_PLAN.md").read_bytes()
    if (root / "config.json").is_file():
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
    commit = api("POST", "/git/commits", json={"message": "Experiment 1.7.1 Hunter PBS/ROCm port and verification", "tree": tree, "parents": [head]})["sha"]
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


def best(root):
    checkpoint = pointer(root, "best")
    step = json.loads((checkpoint / "state.json").read_text())["step"]
    if step <= 0:
        raise ValueError("No trained best checkpoint")
    paths = {f"best/{path.name}": path for path in checkpoint.glob("weights-*.safetensors")}
    expected_shards = json.loads((checkpoint / "config.json").read_text())["tp"]
    if set(paths) != {f"best/weights-{rank:02d}.safetensors" for rank in range(expected_shards)}:
        raise ValueError("Inference shards differ from configured TP")
    paths.update({f"best/{name}": checkpoint / name for name in ("config.json", "state.json")})
    paths["README.md"] = SOURCE / "MODEL_CARD.md"
    api = HfApi(token=credential("HF_TOKEN"))
    if api.whoami()["name"] != "YL95":
        raise ValueError("Wrong HF account")
    api.create_repo(HF, repo_type="model", private=False, exist_ok=True)
    if api.model_info(HF).private:
        raise ValueError("Expected public inference destination")
    commit = api.create_commit(HF, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                                              for name, path in paths.items()],
                               commit_message=f"Experiment 1.7 best inference checkpoint step {step}", num_threads=8)
    remote = {entry.rfilename: entry for entry in api.model_info(HF, revision=commit.oid, files_metadata=True).siblings}
    for name, path in paths.items():
        entry = remote[name]
        expected = digest(path) if entry.lfs is not None else git_blob_file(path)
        actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        if entry.size != path.stat().st_size or actual != expected:
            raise ValueError(f"HF best checkpoint size/digest mismatch: {name}")
    write_json(root / "reports" / "best_publication.json", {"repository": HF, "commit": commit.oid,
               "step": step, "checkpoint": checkpoint.name, "verified_at": datetime.now().astimezone().isoformat()})
    print(json.dumps({"repository": HF, "commit": commit.oid, "best_step": step}), flush=True)


def private_backup(root):
    destination = backup_root() / root.name
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
    report = {"checkpoints": selected, "verified": True, "storage": "private shared-scratch second copy",
              "independent_failure_domain": False, "verified_at": datetime.now().astimezone().isoformat()}
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
        for shard in range(json.loads((latest / "config.json").read_text())["tp"]):
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


def public_snapshot(workspace):
    roots = sorted((workspace / "artifacts").glob("job-*"), key=lambda path: path.stat().st_mtime, reverse=True)
    root = next((path for path in roots if (path / "reports" / "preflight.json").is_file()), None)
    if root is None:
        return {"files": {}}
    files = {}
    for label in ("best", "latest"):
        pointer_path = root / "exports" / f"{label}.json"
        if not pointer_path.exists():
            continue
        directory = root / "exports" / json.loads(pointer_path.read_text())["directory"]
        directory.resolve().relative_to((root / "exports").resolve())
        if (directory / "COMPLETE").read_text().strip() != digest(directory / "manifest.json"):
            raise ValueError("Inference export manifest is incomplete")
        manifest = json.loads((directory / "manifest.json").read_text())
        tp = json.loads((directory / "config.json").read_text())["tp"]
        allowed = {"config.json", "state.json"} | {f"weights-{rank:02d}.safetensors" for rank in range(tp)}
        if set(manifest) != allowed:
            raise ValueError("Unexpected private file in inference export")
        for name, metadata in manifest.items():
            files[f"{label}/{name}"] = {**metadata, "path": (directory / name).relative_to(workspace).as_posix()}
    aggregates = [(f"runs/{root.name}/{path.name}", path) for path in (root / "runs").glob("events.out.tfevents.*")]
    for name in ("preflight.json", "warm_start.json", "private_backup.json"):
        aggregates.append((f"reports/{root.name}/{name}", root / "reports" / name))
    for name in ("history.jsonl", "validation.jsonl", "TRAINING_DONE.json"):
        aggregates.append((f"reports/{root.name}/{name}", root / name))
    for name, path in aggregates:
        if path.is_file():
            files[name] = {"path": path.relative_to(workspace).as_posix(),
                           "bytes": path.stat().st_size, "sha256": digest(path)}
    return {"job": root.name, "files": files}


def serve(workspace, socket_path):
    import socket
    import socketserver
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_UNIX

        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "localhost", 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            snapshot = public_snapshot(workspace)
            if self.path == "/snapshot":
                payload = json.dumps(snapshot).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            relative = unquote(self.path.removeprefix("/files/"))
            allowed = {entry["path"] for entry in snapshot["files"].values()}
            if not self.path.startswith("/files/") or relative not in allowed:
                self.send_error(404)
                return
            path = (workspace / relative).resolve()
            path.relative_to(workspace.resolve())
            with path.open("rb") as stream:
                self.send_response(200)
                self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)

    if socket_path.exists():
        raise ValueError("Publication socket already exists")
    with Server(str(socket_path), Handler) as server:
        socket_path.chmod(0o600)
        print("Private publication socket ready", flush=True)
        server.serve_forever()


def relay(root, endpoint, once=False):
    import threading
    from urllib.parse import quote
    api = HfApi(token=credential("HF_TOKEN"))
    if api.whoami()["name"] != "YL95":
        raise ValueError("Wrong HF account")
    api.create_repo(HF, repo_type="model", private=False, exist_ok=True)
    if api.model_info(HF).private:
        raise ValueError("Unexpected destination visibility")
    wake = threading.Event()
    root.mkdir(parents=True, exist_ok=True)
    previous = {}
    while True:
        try:
            response = requests.get(endpoint + "/snapshot", timeout=60)
            response.raise_for_status()
            snapshot = response.json()
            files = snapshot["files"]
            current = {name: entry["sha256"] for name, entry in files.items()}
            if files and current != previous:
                paths = {"README.md": SOURCE / "MODEL_CARD.md"}
                for name, entry in files.items():
                    path = (root / name).resolve()
                    path.relative_to(root.resolve())
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                        temporary = path.with_name(path.name + ".partial")
                        with requests.get(endpoint + "/files/" + quote(entry["path"]), stream=True, timeout=(30, 120)) as response:
                            response.raise_for_status()
                            with temporary.open("wb") as output:
                                for block in response.iter_content(8 * 1024 * 1024):
                                    output.write(block)
                        if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                            raise ValueError("Export changed during transfer; retry on the next cycle")
                        temporary.replace(path)
                    paths[name] = path
                commit = api.create_commit(HF, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                                                          for name, path in paths.items()],
                                           commit_message=f"Verified Hunter checkpoint and TensorBoard: {snapshot['job']}", num_threads=4)
                remote = {entry.rfilename: entry for entry in api.model_info(HF, revision=commit.oid, files_metadata=True).siblings}
                for name, path in paths.items():
                    entry = remote[name]
                    expected = digest(path) if entry.lfs is not None else git_blob_file(path)
                    actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
                    if entry.size != path.stat().st_size or actual != expected:
                        raise ValueError(f"Remote publication verification failed: {name}")
                previous = current
                report = {"repository": HF, "commit": commit.oid, "job": snapshot["job"],
                          "verified": True, "files": len(paths), "time": datetime.now().astimezone().isoformat()}
                write_json(root / "publication.json", report)
                print(json.dumps(report), flush=True)
        except Exception as error:
            if once:
                raise
            print(json.dumps({"event": "publication_retry", "error_type": type(error).__name__}), flush=True)
        if once:
            return
        wake.wait(300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("source", "backup", "model", "serve", "relay"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18767")
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.credentials:
        from dotenv import load_dotenv
        load_dotenv(arguments.credentials, override=True)
    if arguments.action == "serve":
        serve(arguments.root, arguments.socket)
    elif arguments.action == "relay":
        relay(arguments.root, arguments.endpoint, arguments.once)
    else:
        {"source": source, "backup": private_backup, "model": model}[arguments.action](arguments.root)