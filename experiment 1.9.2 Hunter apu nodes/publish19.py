import argparse
import base64
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import quote

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
import requests
from tqdm.auto import tqdm

from ranker.data import write_json
from ranker.publish import git_blob, git_blob_file
from ranking17 import publish as legacy
from ranking17.recovery import digest, verify


SOURCE = Path(__file__).resolve().parent
GITHUB = "YLiu95/MSc-new-experiments"
FOLDER = "experiment 1.9 Hunter apu nodes"
HF = "YL95/experiment-1.9-hunter-apu-nodes"
RECOVERY_HF = "YL95/experiment-1.9-hunter-recovery"
BASE_SNAPSHOT = legacy.public_snapshot


def public_snapshot(workspace, job=None):
    if job is None:
        snapshot = BASE_SNAPSHOT(workspace)
    else:
        if not re.fullmatch(r"job-\d+\.hunter-pbs01", job):
            raise ValueError("Invalid experiment job")
        root = workspace / "artifacts" / job
        if not any((root / "reports" / name).is_file() for name in ("preflight.json", "migration.json")):
            raise ValueError("Job has no preflight or migration evidence")
        snapshot = {"job": job, "files": {}}
        for label in ("best", "latest"):
            directory = root / "exports" / json.loads((root / "exports" / f"{label}.json").read_text())["directory"]
            directory.resolve().relative_to((root / "exports").resolve())
            if (directory / "COMPLETE").read_text().strip() != digest(directory / "manifest.json"):
                raise ValueError("Incomplete inference export")
            manifest = json.loads((directory / "manifest.json").read_text())
            if set(manifest) != {"config.json", "state.json", "weights-00.safetensors"}:
                raise ValueError("Unexpected private file or topology in inference export")
            for name, metadata in manifest.items():
                snapshot["files"][f"{label}/{name}"] = {
                    **metadata, "path": (directory / name).relative_to(workspace).as_posix()}
        aggregates = [(f"runs/{job}/{path.name}", path) for path in (root / "runs").glob("events.out.tfevents.*")]
        for name in ("preflight.json", "migration.json"):
            aggregates.append((f"reports/{job}/{name}", root / "reports" / name))
        for name in ("history.jsonl", "validation.jsonl", "TRAINING_DONE.json"):
            aggregates.append((f"reports/{job}/{name}", root / name))
        for name, path in aggregates:
            if path.is_file():
                snapshot["files"][name] = {"path": path.relative_to(workspace).as_posix(),
                                           "bytes": path.stat().st_size, "sha256": digest(path)}
    if not snapshot.get("job"):
        return snapshot
    root = workspace / "artifacts" / snapshot["job"]
    additions = {
        "source/runtime-source.tar.gz": workspace / "provenance" / "runtime-source.tar.gz",
        "source/deployed-source.tar.gz": workspace / "source-fixed.tar.gz",
        f"reports/{root.name}/config.json": root / "config.json",
        f"reports/{root.name}/checkpoint_verification.json": root / "reports" / "checkpoint_verification.json",
        f"reports/{root.name}/scheduler.json": root / "reports" / "scheduler.json",
        f"reports/{root.name}/final.json": root / "reports" / "final.json",
    }
    source = workspace / ("source" if root.name == "job-286464.hunter-pbs01" else
                          "source-resume-" + root.name.removeprefix("job-").split(".")[0])
    restarts = sorted((root / "reports").glob("restart-*.json"),
                      key=lambda path: int(path.stem.removeprefix("restart-")))
    if restarts:
        restarted_source = workspace / ("source-restart-" + root.name.removeprefix("job-").split(".")[0])
        report = json.loads(restarts[-1].read_text())
        if report["event"] != "restart_restored" or digest(restarted_source / "train19.py") != report["source_sha256"]:
            raise ValueError("Restart source does not match the recorded running trainer")
        for name in ("train19.py", "run.sh"):
            additions[f"source/{root.name}/before-restart/{name}"] = source / name
        for path in restarts:
            additions[f"reports/{root.name}/{path.name}"] = path
        additions["source/restart-source.tar.gz"] = workspace / "restart-validation.tar.gz"
        source = restarted_source
    for name in ("train19.py", "run.sh"):
        additions[f"source/{root.name}/{name}"] = source / name
    for name, path in additions.items():
        if path.is_file():
            snapshot["files"][name] = {"path": path.relative_to(workspace).as_posix(),
                                       "bytes": path.stat().st_size, "sha256": digest(path)}
    return snapshot


def recovery_snapshot(workspace, job):
    if not re.fullmatch(r"job-\d+\.hunter-pbs01", job):
        raise ValueError("Invalid experiment job")
    root = workspace / "artifacts" / job
    if json.loads((root / "reports" / "scheduler.json").read_text())["job_state"] != "F":
        raise ValueError("Private final backup requires a finished allocation")
    paths = {root / name for name in ("config.json", "latest.json", "best.json")}
    paths.update(root / "reports" / name for name in ("scheduler.json", "final.json", "migration.json"))
    paths.update((root / "reports").glob("restart-*.json"))
    for label in ("best", "latest"):
        directory = (root / json.loads((root / f"{label}.json").read_text())["directory"]).resolve()
        directory.relative_to((root / "checkpoints").resolve())
        verify(directory, workers=4)
        state = json.loads((directory / "state.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        if state["world"] not in (4, 8) or config["tp"] != 1 or config["dp"] != state["world"]:
            raise ValueError("Unexpected recovery topology")
        manifest = json.loads((directory / "manifest.json").read_text())
        required = {"recovery-00.pt", "weights-00.safetensors", "sampler.sqlite", "config.json", "state.json"}
        required.update(f"rng-{rank:04d}.pt" for rank in range(state["world"]))
        if set(manifest) != required:
            raise ValueError("Recovery checkpoint is incomplete or contains unexpected files")
        paths.update(directory / name for name in required | {"manifest.json", "COMPLETE"})
    snapshot = {"job": job, "private_recovery": True,
                "frozen_at": datetime.now().astimezone().isoformat(), "files": {}}
    for path in sorted(paths):
        if path.is_file():
            path.resolve().relative_to(root.resolve())
            snapshot["files"][f"{job}/{path.relative_to(root).as_posix()}"] = {
                "path": path.relative_to(workspace).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
    return snapshot


def freeze_snapshot(workspace, snapshot):
    parent = workspace / "publication"
    parent.mkdir(parents=True, exist_ok=True)
    frozen = Path(tempfile.mkdtemp(prefix=snapshot["job"] + "-", dir=parent))
    result = {"job": snapshot["job"], "files": {}, "frozen_at": datetime.now().astimezone().isoformat()}
    try:
        for name, entry in snapshot["files"].items():
            source = (workspace / entry["path"]).resolve()
            source.relative_to(workspace.resolve())
            target = frozen / name
            target.resolve().relative_to(frozen.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith(("best/", "latest/")):
                os.link(source, target)
                if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
                    raise ValueError("Checkpoint changed during snapshot; retry after save finishes")
            else:
                shutil.copyfile(source, target)
            result["files"][name] = {"path": target.relative_to(workspace).as_posix(),
                                     "bytes": target.stat().st_size, "sha256": digest(target)}
        write_json(frozen / "snapshot.json", result)
        return result
    except BaseException:
        shutil.rmtree(frozen)
        raise


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


def snapshot_at(endpoint, job="job-286464.hunter-pbs01", private_recovery=False):
    response = requests.get(endpoint + "/snapshot", timeout=60)
    response.raise_for_status()
    snapshot = response.json()
    if snapshot.get("job") != job:
        raise ValueError("Unexpected experiment job at publication endpoint")
    if bool(snapshot.get("private_recovery")) != private_recovery:
        raise ValueError("Snapshot privacy does not match the requested publication")
    return snapshot


def download_with_progress(root, name, entry, endpoint):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]:
        print(json.dumps({"event": "cache_verified", "file": name}), flush=True)
        return path
    temporary = path.with_name(path.name + ".partial")
    temporary.unlink(missing_ok=True)
    if shutil.disk_usage(root).free < entry["bytes"] + 2 * 1024**3:
        raise RuntimeError("Insufficient free disk space including 2 GiB reserve")
    with requests.get(endpoint + "/files/" + quote(entry["path"]), stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output, tqdm(total=entry["bytes"], desc=f"Hunter -> PC: {name}",
                                                  unit="B", unit_scale=True, mininterval=1) as progress:
            for block in response.iter_content(8 * 1024 * 1024):
                output.write(block)
                progress.update(len(block))
    print(json.dumps({"event": "verifying_download", "file": name}), flush=True)
    if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
        raise ValueError("Downloaded file failed size/SHA-256 verification")
    temporary.replace(path)
    return path


def publish_source(root, endpoint, job="job-286464.hunter-pbs01"):
    snapshot = snapshot_at(endpoint, job)
    archive = legacy.cache_public_file(root, "source/runtime-source.tar.gz",
                                      snapshot["files"]["source/runtime-source.tar.gz"], endpoint)
    files = runtime_files(archive)
    for name in ("slice_model.py", "train19.py", "run.sh", "publish19.py", "cloud_cleanup.py", "README.md", "MODEL_CARD.md",
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


def verify_cloud_files(api, repository, revision, paths, private):
    info = api.model_info(repository, revision=revision, files_metadata=True)
    if info.private != private:
        raise ValueError("Cloud repository privacy changed")
    remote = {entry.rfilename: entry for entry in info.siblings}
    verified = {}
    for name, path in paths.items():
        entry = remote[name]
        actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        expected = digest(path) if entry.lfs is not None else git_blob_file(path)
        if entry.size != path.stat().st_size or actual != expected:
            raise ValueError(f"Remote file verification failed: {name}")
        verified[name] = {"bytes": entry.size, "digest": actual}
    return verified


def publish_recovery(root, endpoint, job):
    api = HfApi(token=legacy.credential("HF_TOKEN"))
    if api.whoami()["name"] != "YL95":
        raise ValueError("Wrong Hugging Face account")
    api.create_repo(RECOVERY_HF, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(RECOVERY_HF).private:
        raise ValueError("Recovery destination must be private")
    snapshot = snapshot_at(endpoint, job, private_recovery=True)
    write_json(root / "snapshot.json", snapshot)
    paths = {name: download_with_progress(root, name, entry, endpoint)
             for name, entry in snapshot["files"].items()}
    if not paths or not any(name.endswith("/recovery-00.pt") for name in paths):
        raise ValueError("No full recovery payload in snapshot")
    if not api.model_info(RECOVERY_HF).private:
        raise ValueError("Recovery destination must remain private")
    print(json.dumps({"event": "uploading_private_recovery", "repository": RECOVERY_HF,
                      "files": len(paths)}), flush=True)
    commit = api.create_commit(RECOVERY_HF, operations=[
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in paths.items()],
        commit_message=f"Verified full best/latest recovery for {job}", num_threads=4)
    verified = verify_cloud_files(api, RECOVERY_HF, commit.oid, paths, private=True)
    report = {"repository": RECOVERY_HF, "commit": commit.oid, "job": job, "private": True,
              "verified": True, "files": verified, "time": datetime.now().astimezone().isoformat()}
    write_json(root / "recovery-publication.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)


def publish_model(root, endpoint, metrics_only=False, job="job-286464.hunter-pbs01"):
    api = HfApi(token=legacy.credential("HF_TOKEN"))
    if api.whoami()["name"] != "YL95":
        raise ValueError("Wrong Hugging Face account")
    api.create_repo(HF, repo_type="model", private=False, exist_ok=True)
    if api.model_info(HF).private:
        raise ValueError("Expected public inference repository")
    snapshot = snapshot_at(endpoint, job)
    if job != "job-286464.hunter-pbs01" and not snapshot.get("frozen_at"):
        raise ValueError("Continuation publication requires a frozen snapshot")
    write_json(root / "snapshot.json", snapshot)
    print(json.dumps({"event": "backup_started", "job": job, "frozen_at": snapshot.get("frozen_at"),
                      "files": len(snapshot["files"]), "repository": HF}), flush=True)
    paths = {"README.md": SOURCE / "MODEL_CARD.md"}
    cached = {}
    for name, entry in snapshot["files"].items():
        if metrics_only and name.startswith(("best/", "latest/", "source/")):
            continue
        print(json.dumps({"event": "download", "file": name, "bytes": entry["bytes"]}), flush=True)
        if entry["sha256"] in cached:
            paths[name] = cached[entry["sha256"]]
            print(json.dumps({"event": "reuse_identical_content", "file": name}), flush=True)
        else:
            paths[name] = download_with_progress(root, name, entry, endpoint)
            cached[entry["sha256"]] = paths[name]
    if not metrics_only and not {"best/weights-00.safetensors", "latest/weights-00.safetensors"} <= paths.keys():
        raise ValueError("Both best and latest inference checkpoints are required")
    states = {label: json.loads(paths[f"{label}/state.json"].read_text())
              for label in ("best", "latest") if f"{label}/state.json" in paths}
    if states:
        summary = {"job": job, "frozen_at": snapshot.get("frozen_at"), "checkpoints": states,
                   "files": {name: {key: entry[key] for key in ("bytes", "sha256")}
                             for name, entry in snapshot["files"].items()}}
        summary_name = f"reports/{job}/publication-snapshot.json"
        write_json(root / summary_name, summary)
        paths[summary_name] = root / summary_name
        card = (SOURCE / "MODEL_CARD.md").read_text(encoding="utf-8")
        card += (f"\n## Published Snapshot\n\nJob: `{job}`. Snapshot time: `{snapshot.get('frozen_at', 'original run')}`.\n\n"
                 f"Best saved step: **{states['best']['step']}**. Latest saved step: **{states['latest']['step']}**.\n")
        final_path = paths.get(f"reports/{job}/final.json")
        if final_path is not None:
            final = json.loads(final_path.read_text())
            card += (f"Scheduler state: **{final['scheduler_state']}**. Last completed update: **{final['last_update']}**. "
                     f"Updates after the latest saved checkpoint: **{final['unsaved_updates']}**, not recoverable.\n")
        else:
            card += "This is a point-in-time backup of completed checkpoints, not a claim that training has finished.\n"
        (root / "README.md").write_text(card, encoding="utf-8")
        paths["README.md"] = root / "README.md"
    print(json.dumps({"event": "uploading_to_huggingface", "job": job, "checkpoints": states,
                      "files": len(paths)}), flush=True)
    commit = api.create_commit(HF, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                                              for name, path in paths.items()],
                               commit_message="Experiment 1.9 TensorBoard and metrics" if metrics_only else
                               "Verified Experiment 1.9 best/latest inference checkpoints and metrics", num_threads=4)
    print(json.dumps({"event": "verifying_cloud_commit", "commit": commit.oid}), flush=True)
    verified = verify_cloud_files(api, HF, commit.oid, paths, private=False)
    report = {"repository": HF, "commit": commit.oid, "job": snapshot["job"], "verified": True,
              "metrics_only": metrics_only, "files": verified, "checkpoints": states,
              "frozen_at": snapshot.get("frozen_at"), "time": datetime.now().astimezone().isoformat()}
    write_json(root / ("metrics.json" if metrics_only else "publication.json"), report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "source", "model", "recovery"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18767")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--job", default="job-286464.hunter-pbs01")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--private-recovery", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "serve":
        if arguments.private_recovery:
            frozen = recovery_snapshot(arguments.root, arguments.job)
            print(json.dumps({"event": "private_recovery_ready", "files": len(frozen["files"]),
                              "bytes": sum(entry["bytes"] for entry in frozen["files"].values())}), flush=True)
            legacy.public_snapshot = lambda workspace: frozen
        elif arguments.freeze:
            frozen = freeze_snapshot(arguments.root, public_snapshot(arguments.root, arguments.job))
            print(json.dumps({"event": "snapshot_frozen", "job": frozen["job"],
                              "files": len(frozen["files"]), "frozen_at": frozen["frozen_at"]}), flush=True)
            legacy.public_snapshot = lambda workspace: frozen
        else:
            legacy.public_snapshot = lambda workspace: public_snapshot(workspace, arguments.job)
        legacy.serve(arguments.root, arguments.socket)
    else:
        if arguments.credentials:
            load_dotenv(arguments.credentials)
        if arguments.action == "source":
            publish_source(arguments.root, arguments.endpoint, arguments.job)
        elif arguments.action == "recovery":
            publish_recovery(arguments.root, arguments.endpoint, arguments.job)
        else:
            publish_model(arguments.root, arguments.endpoint, arguments.metrics_only, arguments.job)