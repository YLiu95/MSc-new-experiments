import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import socketserver
import subprocess
import tarfile
import threading
import time
from urllib.parse import unquote

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
import requests

import publish19 as shared
from ranker.data import write_json
from ranker.publish import git_blob
from ranking17.recovery import digest, verify
from policy192 import JOB as ALLOCATION_JOB, WORLD


JOB = f"job-{ALLOCATION_JOB}"
HF = "YL95/experiment-1.9.2-hunter-apu-nodes"
PRIVATE_HF = "YL95/experiment-1.9.2-hunter-recovery"
SOURCE = Path(__file__).resolve().parent
INTERVAL = 300


def status(base):
    root = base / "artifacts" / JOB
    path = root / "control/status.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result.update(job=JOB, experiment="1.9.2", finished=(root / "EXPERIMENT_DONE.json").is_file())
    result["failure"] = json.loads((root / "control/failure.json").read_text()) if (root / "control/failure.json").exists() else None
    result["allocation_ended"] = (root / "control/allocation-ended.json").exists()
    result["finished"] = result["finished"] or result["allocation_ended"]
    return result


def collect(base, mode):
    if mode not in ("metrics", "exports", "private"):
        raise ValueError("Unknown publication mode")
    root = base / "artifacts" / JOB
    snapshot = {"job": JOB, "experiment": "1.9.2", "private_recovery": mode == "private", "files": {}}

    def add(name, path):
        if path.is_file():
            path.resolve().relative_to(base.resolve())
            snapshot["files"][name] = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}

    for arm in ("A", "B"):
        directory = root / f"arm-{arm}"
        if mode in ("exports", "private"):
            prefix = directory / "exports" if mode == "exports" else directory
            if mode == "private" and not all((root / f"arm-{item}/TRAINING_DONE.json").exists() for item in ("A", "B")):
                raise ValueError("Private final recovery requires both stopped arms")
            for label in ("best", "latest"):
                pointer = prefix / f"{label}.json"
                if not pointer.exists():
                    continue
                checkpoint = (prefix / json.loads(pointer.read_text())["directory"]).resolve()
                checkpoint.relative_to(prefix.resolve())
                manifest = verify(checkpoint, workers=4)
                expected = {"config.json", "state.json", "weights-00.safetensors"}
                if mode == "private":
                    checkpoint_world = json.loads((checkpoint / "state.json").read_text())["world"]
                    if checkpoint_world not in (12, WORLD):
                        raise ValueError("Unexpected recovery worker layout")
                    expected |= {"sampler.sqlite", "recovery-00.pt"} | {f"rng-{rank:04d}.pt" for rank in range(checkpoint_world)}
                if set(manifest) != expected:
                    raise ValueError("Unexpected checkpoint contents or privacy boundary")
                for name in expected | {"manifest.json", "COMPLETE"}:
                    add(f"arm-{arm}/{label}/{name}", checkpoint / name)
        if mode in ("exports", "private"):
            continue
        for name in ("history.jsonl", "validation.jsonl", "TRAINING_DONE.json", "config.json", "best.json", "latest.json"):
            add(f"reports/{JOB}/arm-{arm}/{name}", directory / name)
        for name in ("preflight.json", "baseline.json", "protocol.json", "checkpoint_verification.json", "resume.json"):
            add(f"reports/{JOB}/arm-{arm}/{name}", directory / "reports" / name)
        for path in (directory / "logs").rglob("*"):
            if path.name in ("stdout.log", "stderr.log", "error.json"):
                add(f"reports/{JOB}/arm-{arm}/logs/{path.relative_to(directory / 'logs').as_posix()}", path)
        for path in (directory / "runs").rglob("events.out.tfevents.*"):
            add(f"runs/{path.relative_to(directory / 'runs').as_posix()}", path)
    if mode == "metrics":
        for name in ("job.log", "final-evaluation.log", "EXPERIMENT_DONE.json", "RESULTS.md"):
            add(f"reports/{JOB}/{name}", root / name)
        for name in ("watch.log", "status.json", "deadlines.json", "launch.json", "failure.json", "launch.log", "allocation-ended.json", "batch-exit.json"):
            add(f"reports/{JOB}/control/{name}", root / "control" / name)
        for name in ("exposure.json", "universe.json"):
            add(f"reports/{JOB}/audits/{name}", root / "audits" / name)
        add(f"reports/{JOB}/final-test/results.json", root / "final-test/results.json")
        for path in (root / "runs").rglob("events.out.tfevents.*"):
            add(f"runs/{path.relative_to(root / 'runs').as_posix()}", path)
        for path in (base / "reference-original").rglob("events.out.tfevents.*"):
            add(f"reference-original/{path.relative_to(base / 'reference-original').as_posix()}", path)
    return snapshot


def serve(base, socket_path):
    snapshots, temporary = {}, {}

    class Server(socketserver.ThreadingMixIn, HTTPServer):
        address_family = socket.AF_UNIX
        daemon_threads = True

        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "localhost", 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            mode = self.path.strip("/").split("/")[0]
            if mode not in ("metrics", "exports", "private") or self.path != f"/{mode}/release":
                self.send_error(404)
                return
            if mode in temporary:
                shutil.rmtree(temporary.pop(mode))
            snapshots.pop(mode, None)
            self.reply({"released": mode})

        def do_GET(self):
            if self.path == "/status":
                self.reply(status(base))
                return
            parts = self.path.strip("/").split("/", 2)
            if len(parts) < 2 or parts[0] not in ("metrics", "exports", "private"):
                self.send_error(404)
                return
            mode, action = parts[:2]
            if action == "snapshot":
                if mode not in snapshots:
                    original = collect(base, mode)
                    snapshot = shared.freeze_snapshot(base, original)
                    snapshot.update(experiment="1.9.2", private_recovery=mode == "private")
                    snapshots[mode] = snapshot
                    if snapshot["files"]:
                        relative = next(iter(snapshot["files"].values()))["path"]
                        temporary[mode] = base.joinpath(*PurePosixPath(relative).parts[:2])
                self.reply(snapshots[mode])
                return
            relative = unquote(parts[2]) if len(parts) == 3 else ""
            allowed = {entry["path"] for entry in snapshots.get(mode, {}).get("files", {}).values()}
            if action != "files" or relative not in allowed:
                self.send_error(404)
                return
            path = (base / relative).resolve()
            path.relative_to(base.resolve())
            with path.open("rb") as stream:
                self.send_response(200)
                self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)

    if socket_path.exists():
        raise ValueError("Publication socket already exists")
    with Server(str(socket_path), Handler) as server:
        socket_path.chmod(0o600)
        print(json.dumps({"event": "publisher_ready", "job": JOB}), flush=True)
        server.serve_forever()


def publish_once(root, endpoint, mode):
    current = requests.get(endpoint + "/status", timeout=30)
    current.raise_for_status()
    current = current.json()
    if current.get("job") != JOB or current.get("experiment") != "1.9.2":
        raise ValueError("Wrong experiment at publication endpoint")
    route = endpoint + "/" + mode
    requests.post(route + "/release", timeout=60).raise_for_status()
    snapshot = shared.snapshot_at(route, JOB, private_recovery=mode == "private")
    if snapshot.get("experiment") != "1.9.2" or not snapshot.get("frozen_at"):
        raise ValueError("Expected an immutable Experiment 1.9.2 snapshot")
    if mode == "exports" and not all(f"arm-{arm}/{label}/weights-00.safetensors" in snapshot["files"] for arm in ("A", "B") for label in ("best", "latest")):
        requests.post(route + "/release", timeout=60).raise_for_status()
        return current["finished"]
    paths = {}
    if mode != "private":
        paths["README.md"] = SOURCE / "MODEL_CARD.md"
    for name, entry in snapshot["files"].items():
        paths[name] = shared.download_with_progress(root, name, entry, route)
    api = HfApi(token=shared.legacy.credential("HF_TOKEN"))
    private = mode == "private"
    repository = PRIVATE_HF if private else HF
    if api.whoami()["name"] != "YL95":
        raise ValueError("Unexpected Hugging Face account")
    api.create_repo(repository, repo_type="model", private=private, exist_ok=True)
    if api.model_info(repository).private != private:
        raise ValueError("Repository privacy differs from publication mode")
    commit = api.create_commit(repository, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in paths.items()],
                               commit_message=f"Experiment 1.9.2 {mode}: both arms, {JOB}", num_threads=4)
    verified = shared.verify_cloud_files(api, repository, commit.oid, paths, private=private)
    receipt = {"repository": repository, "job": JOB, "mode": mode, "commit": commit.oid, "verified": True,
               "private": private, "files": verified, "frozen_at": snapshot["frozen_at"], "time": time.time()}
    write_json(root / f"{mode}.json", receipt)
    write_json(root / "backup-status.json", {"ok": True, "mode": mode, "commit": commit.oid, "time": receipt["time"]})
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}), flush=True)
    requests.post(route + "/release", timeout=60).raise_for_status()
    return current["finished"]


def relay(root, endpoint, mode, once=False):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f"{mode}-relay.lock"
    lock.mkdir()
    try:
        while True:
            started, failed = time.monotonic(), False
            try:
                finished = publish_once(root, endpoint, mode)
                if once or finished:
                    return
            except Exception as error:
                write_json(root / "backup-status.json", {"ok": False, "error": str(error), "time": time.time()})
                print(json.dumps({"event": "backup_retry", "error": str(error), "mode": mode}), flush=True)
                if once:
                    raise
                failed = True
            threading.Event().wait(max(0, (60 if failed else INTERVAL if mode == "metrics" else 1800) - (time.monotonic() - started)))
    finally:
        lock.rmdir()


def notify(endpoint, phase="allocation"):
    if phase not in ("allocation", "training", "completion"):
        raise ValueError("Unknown notification phase")
    previous = None
    while True:
        try:
            response = requests.get(endpoint + "/status", timeout=30)
            response.raise_for_status()
            current = response.json()
        except requests.RequestException as error:
            print(json.dumps({"event": "notification_connection_retry", "error": str(error), "time": time.time()}), flush=True)
            threading.Event().wait(30)
            continue
        if current.get("job") != JOB or current.get("experiment") != "1.9.2":
            raise ValueError("Wrong experiment at notification endpoint")
        if current.get("failure"):
            raise RuntimeError(json.dumps(current["failure"]))
        if not current.get("finished") and time.time() - current.get("time", 0) > 180:
            raise RuntimeError("Hunter watcher heartbeat is more than three minutes old")
        arms = current.get("arms", {})
        healthy = all(arms.get(arm, {}).get("preflight_passed") and
                      arms.get(arm, {}).get("initial_checkpoint_complete") and
                      arms.get(arm, {}).get("new_updates", 0) >= 3 for arm in ("A", "B"))
        if current.get("finished") or (phase == "allocation" and current.get("state") == "R") or (phase == "training" and healthy):
            if phase == "training" and not healthy:
                raise RuntimeError("Allocation ended before both arms passed startup: " + json.dumps(current))
            print(json.dumps({"event": "CURRENT_SESSION_NOTIFICATION", "phase": phase, **current}), flush=True)
            return current
        fingerprint = (current.get("state"), tuple(arms.get(arm, {}).get("new_updates", 0) for arm in ("A", "B")))
        if fingerprint != previous:
            print(json.dumps({"event": "monitor_wait", "phase": phase, "job": JOB, "state": current.get("state"), "arms": arms}), flush=True)
            previous = fingerprint
        threading.Event().wait(30)


def publish_source(root, archive_path):
    files = {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or name.suffix not in (".py", ".sh", ".md", ".json"):
                raise ValueError("Unexpected source archive entry")
            files[member.name] = archive.extractfile(member).read()
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + shared.legacy.credential("GITHUB_TOKEN"), "Accept": "application/vnd.github+json"})
    repository, folder = "YLiu95/MSc-new-experiments", "experiment 1.9.2 Hunter apu nodes"
    def api(method, path, **kwargs):
        response = session.request(method, f"https://api.github.com/repos/{repository}{path}", timeout=60, **kwargs)
        if not response.ok:
            raise RuntimeError(f"GitHub HTTP {response.status_code}")
        return response.json()
    head = api("GET", "/git/ref/heads/main")["object"]["sha"]
    tree = api("GET", f"/git/commits/{head}")["tree"]["sha"]
    entries = []
    for name, content in files.items():
        blob = api("POST", "/git/blobs", json={"content": base64.b64encode(content).decode(), "encoding": "base64"})["sha"]
        if blob != git_blob(content):
            raise ValueError("Source blob digest mismatch")
        entries.append({"path": f"{folder}/{name}", "mode": "100755" if name.endswith(".sh") else "100644", "type": "blob", "sha": blob})
    tree = api("POST", "/git/trees", json={"base_tree": tree, "tree": entries})["sha"]
    commit = api("POST", "/git/commits", json={"message": "Back up Experiment 1.9.2 two-arm Hunter training", "tree": tree, "parents": [head]})["sha"]
    api("PATCH", "/git/refs/heads/main", json={"sha": commit, "force": False})
    remote = api("GET", f"/git/trees/{tree}?recursive=1")
    mapping = {entry["path"]: entry["sha"] for entry in remote["tree"]}
    if remote.get("truncated") or any(mapping.get(f"{folder}/{name}") != git_blob(content) for name, content in files.items()):
        raise ValueError("Source tree verification failed")
    write_json(root / "github.json", {"repository": repository, "folder": folder, "commit": commit, "verified": True, "files": len(files)})
    print(json.dumps({"event": "source_published", "commit": commit, "files": len(files)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "metrics", "exports", "private", "notify", "source"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18773")
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--phase", choices=("allocation", "training", "completion"), default="allocation")
    arguments = parser.parse_args()
    if arguments.credentials:
        load_dotenv(arguments.credentials)
    if arguments.action == "serve":
        serve(arguments.root, arguments.socket)
    elif arguments.action == "notify":
        notify(arguments.endpoint, arguments.phase)
    elif arguments.action == "source":
        publish_source(arguments.root, arguments.archive)
    else:
        relay(arguments.root, arguments.endpoint, arguments.action, arguments.once)