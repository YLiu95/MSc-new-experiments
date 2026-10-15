import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import socketserver
import tarfile
import threading
import time
from urllib.parse import unquote

from dotenv import load_dotenv
import requests

import publish19 as shared
from ranker.data import write_json
from ranker.publish import git_blob
from ranking17.recovery import digest, verify


SOURCE = Path(__file__).resolve().parent
JOB = "job-286972.hunter-pbs01"
HF = "YL95/experiment-1.9.1-hunter-apu-nodes"
PRIVATE_HF = "YL95/experiment-1.9.1-hunter-recovery"
GITHUB = "YLiu95/MSc-new-experiments"
FOLDER = "experiment 1.9.1 Hunter apu nodes"
shared.HF, shared.RECOVERY_HF, shared.SOURCE = HF, PRIVATE_HF, SOURCE


def status(base):
    root = base / "artifacts" / JOB
    result = {"job": JOB, "latest_step": None, "best_step": None, "finished": False}
    for label in ("best", "latest"):
        pointer = root / "exports" / f"{label}.json"
        if pointer.is_file():
            directory = root / "exports" / json.loads(pointer.read_text())["directory"]
            directory.resolve().relative_to((root / "exports").resolve())
            result[f"{label}_step"] = json.loads((directory / "state.json").read_text())["step"]
    path = root / "control" / "status.json"
    result["scheduler"] = json.loads(path.read_text()) if path.exists() else {}
    result["finished"] = (root / "TRAINING_DONE.json").is_file() or result["scheduler"].get("state") == "F"
    return result


def collect(base, mode):
    if mode not in ("public", "metrics", "private"):
        raise ValueError("Unknown snapshot mode")
    root = base / "artifacts" / JOB
    snapshot = {"job": JOB, "files": {}, "private_recovery": mode == "private"}

    def add(name, path, metadata=None):
        if path.is_file():
            path.resolve().relative_to(base.resolve())
            snapshot["files"][name] = {"path": path.relative_to(base).as_posix(),
                                       **(metadata or {"bytes": path.stat().st_size, "sha256": digest(path)})}

    if mode == "private":
        if not status(base)["finished"]:
            raise ValueError("Full recovery publication requires stopped training")
        for label in ("best", "latest"):
            pointer = root / f"{label}.json"
            directory = (root / json.loads(pointer.read_text())["directory"]).resolve()
            directory.relative_to((root / "checkpoints").resolve())
            manifest = verify(directory, workers=4)
            required = {"config.json", "state.json", "sampler.sqlite", "recovery-00.pt", "weights-00.safetensors"}
            required.update(f"rng-{rank:04d}.pt" for rank in range(48))
            if set(manifest) != required or json.loads((directory / "state.json").read_text())["world"] != 48:
                raise ValueError("Incomplete 48-worker recovery checkpoint")
            for name, metadata in manifest.items():
                path = directory / name
                add(f"{JOB}/{path.relative_to(root).as_posix()}", path, metadata)
            for name in ("manifest.json", "COMPLETE"):
                path = directory / name
                add(f"{JOB}/{path.relative_to(root).as_posix()}", path)
            add(f"{JOB}/{label}.json", pointer)
        add(f"{JOB}/config.json", root / "config.json")
        add(f"{JOB}/TRAINING_DONE.json", root / "TRAINING_DONE.json")
        snapshot["frozen_at"] = datetime.now().astimezone().isoformat()
        return snapshot
    if mode == "public":
        for label in ("best", "latest"):
            pointer = root / "exports" / f"{label}.json"
            if not pointer.is_file():
                continue
            directory = root / "exports" / json.loads(pointer.read_text())["directory"]
            directory.resolve().relative_to((root / "exports").resolve())
            if (directory / "COMPLETE").read_text().strip() != digest(directory / "manifest.json"):
                raise ValueError("Incomplete inference export")
            manifest = json.loads((directory / "manifest.json").read_text())
            if set(manifest) != {"config.json", "state.json", "weights-00.safetensors"}:
                raise ValueError("Private file in public export")
            for name, metadata in manifest.items():
                add(f"{label}/{name}", directory / name, metadata)
        add("source/training-source.tar.gz", base / "training-source.tar.gz")
    for name in ("history.jsonl", "validation.jsonl", "TRAINING_DONE.json", "config.json"):
        add(f"reports/{JOB}/{name}", root / name)
    for name in ("preflight.json", "migration.json", "checkpoint_verification.json"):
        add(f"reports/{JOB}/{name}", root / "reports" / name)
    for name in ("status.json", "deadlines.json", "launch.json", "failure.json"):
        add(f"reports/{JOB}/control/{name}", root / "control" / name)
    for path in (root / "runs").glob("events.out.tfevents.*"):
        add(f"runs/{JOB}/{path.name}", path)
    return snapshot


def serve(base, socket_path):
    snapshots, temporary = {}, []

    class Server(HTTPServer):
        address_family = socket.AF_UNIX

        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "localhost", 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def reply(self, record):
            data = json.dumps(record).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/release":
                self.send_error(404)
                return
            for path in temporary:
                shutil.rmtree(path)
            temporary.clear()
            snapshots.clear()
            self.reply({"released": True})

        def do_GET(self):
            if self.path == "/status":
                self.reply(status(base))
                return
            mode, path = "public", self.path
            for candidate in ("metrics", "private"):
                if path.startswith(f"/{candidate}/"):
                    mode, path = candidate, path.removeprefix(f"/{candidate}")
                    break
            if path == "/snapshot":
                snapshot = collect(base, mode)
                if mode != "private":
                    snapshot = shared.freeze_snapshot(base, snapshot)
                    if snapshot["files"]:
                        relative = next(iter(snapshot["files"].values()))["path"]
                        temporary.append(base.joinpath(*PurePosixPath(relative).parts[:2]))
                snapshots[mode] = snapshot
                self.reply(snapshot)
                return
            relative = unquote(path.removeprefix("/files/"))
            allowed = {entry["path"] for entry in snapshots.get(mode, {}).get("files", {}).values()}
            if not path.startswith("/files/") or relative not in allowed:
                self.send_error(404)
                return
            source = (base / relative).resolve()
            source.relative_to(base.resolve())
            with source.open("rb") as stream:
                self.send_response(200)
                self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)

    if socket_path.exists():
        raise ValueError("Backup socket already exists")
    with Server(str(socket_path), Handler) as server:
        socket_path.chmod(0o600)
        print(json.dumps({"event": "backup_server_ready", "job": JOB}), flush=True)
        server.serve_forever()


def publish_source(root):
    import base64
    files = {}
    with tarfile.open(SOURCE / "training-source.tar.gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or name.suffix not in (".py", ".sh", ".txt"):
                raise ValueError("Unexpected source archive member")
            files[member.name] = archive.extractfile(member).read()
    for name in ("publish191.py", "README.md", "MODEL_CARD.md", "tests/test_experiment191.py"):
        files[name] = (SOURCE / name).read_bytes()
    files["publish19.py"] = Path(shared.__file__).read_bytes()
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + shared.legacy.credential("GITHUB_TOKEN"),
                            "Accept": "application/vnd.github+json"})

    def api(method, path, **kwargs):
        response = session.request(method, f"https://api.github.com/repos/{GITHUB}{path}", timeout=60, **kwargs)
        if not response.ok:
            raise RuntimeError(f"GitHub backup failed: HTTP {response.status_code}")
        return response.json()

    head = api("GET", "/git/ref/heads/main")["object"]["sha"]
    tree = api("GET", f"/git/commits/{head}")["tree"]["sha"]
    entries = []
    for name, content in files.items():
        blob = api("POST", "/git/blobs", json={"content": base64.b64encode(content).decode(), "encoding": "base64"})["sha"]
        if blob != git_blob(content):
            raise ValueError("GitHub blob mismatch")
        entries.append({"path": f"{FOLDER}/{name}", "mode": "100755" if name.endswith(".sh") else "100644",
                        "type": "blob", "sha": blob})
    tree = api("POST", "/git/trees", json={"base_tree": tree, "tree": entries})["sha"]
    commit = api("POST", "/git/commits", json={"message": "Back up Experiment 1.9.1 twelve-node Hunter training",
                                               "tree": tree, "parents": [head]})["sha"]
    api("PATCH", "/git/refs/heads/main", json={"sha": commit, "force": False})
    remote = api("GET", f"/git/trees/{tree}?recursive=1")
    mapping = {entry["path"]: entry["sha"] for entry in remote["tree"]}
    if remote.get("truncated") or any(mapping.get(f"{FOLDER}/{name}") != git_blob(content) for name, content in files.items()):
        raise ValueError("GitHub source verification failed")
    report = {"repository": GITHUB, "folder": FOLDER, "commit": commit, "files": len(files), "verified": True}
    write_json(root / "github.json", report)
    print(json.dumps(report), flush=True)


def relay(root, endpoint):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "relay.lock"
    lock.mkdir()
    last_full, last_steps = 0, None
    try:
        while True:
            try:
                response = requests.get(endpoint + "/status", timeout=60)
                response.raise_for_status()
                current = response.json()
                print(json.dumps({"event": "remote_status", **current}), flush=True)
                requests.post(endpoint + "/release", timeout=60).raise_for_status()
                steps = (current["best_step"], current["latest_step"])
                full = all(step is not None for step in steps) and steps != last_steps and (
                    time.time() - last_full >= 600 or current["finished"])
                shared.publish_model(root, endpoint if full else endpoint + "/metrics", not full, JOB)
                if full:
                    last_full, last_steps = time.time(), steps
                if current["finished"] and steps == last_steps:
                    shared.publish_recovery(root / "private", endpoint + "/private", JOB)
                    print(json.dumps({"event": "all_backups_complete", "job": JOB}), flush=True)
                    return
            except Exception as error:
                print(json.dumps({"event": "backup_retry", "error": str(error), "time": time.time()}), flush=True)
            threading.Event().wait(60)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "source", "relay"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18769")
    parser.add_argument("--credentials", type=Path)
    arguments = parser.parse_args()
    if arguments.credentials:
        load_dotenv(arguments.credentials)
    if arguments.action == "serve":
        serve(arguments.root, arguments.socket)
    elif arguments.action == "source":
        publish_source(arguments.root)
    else:
        relay(arguments.root, arguments.endpoint)