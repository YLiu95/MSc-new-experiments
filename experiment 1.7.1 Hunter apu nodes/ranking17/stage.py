import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from urllib.parse import quote


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(root, credentials):
    os.environ["HF_HOME"] = str(root / "cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, snapshot_download
    from ranker.contract import DATASET, REVISION, MARKETS
    from ranking17.warmstart import SOURCE_REPO, SOURCE_REVISION
    token = dotenv_values(credentials)["HF_TOKEN"]
    api = HfApi(token=token)
    if api.whoami()["name"] != "YL95":
        raise ValueError("Unexpected HF account")
    for repository, revision, kind, folder, patterns in (
        (DATASET, REVISION, "dataset", "dataset", [f"data/{market}/*.parquet" for market in MARKETS]),
        (SOURCE_REPO, SOURCE_REVISION, "model", "warmstart", ["latest/*"]),
    ):
        destination = root / folder
        snapshot_download(repository, repo_type=kind, revision=revision, token=token,
                          local_dir=destination, allow_patterns=patterns, max_workers=4)
        info = api.repo_info(repository, repo_type=kind, revision=revision, files_metadata=True)
        remote = {entry.rfilename: entry for entry in info.siblings}
        files = {}
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or ".cache" in path.parts or path.name == "download.json":
                continue
            name = path.relative_to(destination).as_posix()
            entry = remote[name]
            checksum = sha256(path)
            if entry.size != path.stat().st_size or (entry.lfs is not None and entry.lfs.sha256 != checksum):
                raise ValueError(f"Downloaded artifact verification failed: {name}")
            files[name] = {"bytes": path.stat().st_size, "sha256": checksum}
        (destination / "download.json").write_text(json.dumps(
            {"repository": repository, "revision": revision, "files": files}, indent=2) + "\n")
        print(json.dumps({"repository": repository, "files": len(files), "verified": True}), flush=True)


def receive(root, socket):
    for folder in ("wheels", "dataset", "warmstart"):
        destination = root / folder
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        result = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--max-time", "60",
                                          "--unix-socket", str(socket), f"http://localhost/{folder}/download.json"])
        manifest = json.loads(result)
        for name, expected in manifest["files"].items():
            path = (destination / name).resolve()
            path.relative_to(destination.resolve())
            if path.is_file() and path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"]:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--max-time", "1800", "--retry", "2",
                            "--unix-socket", str(socket), "--output", str(temporary),
                            f"http://localhost/{folder}/{quote(name)}"], check=True)
            if temporary.stat().st_size != expected["bytes"] or sha256(temporary) != expected["sha256"]:
                raise ValueError(f"Transferred artifact verification failed: {name}")
            temporary.replace(path)
            print(json.dumps({"received": f"{folder}/{name}", "bytes": expected["bytes"]}), flush=True)
        (destination / "download.json").write_bytes(result)


def bundle(root):
    source = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    wheelhouse = root / "wheels"
    if wheelhouse.is_dir():
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(wheelhouse.glob("*.whl"))}
        (wheelhouse / "download.json").write_text(json.dumps(
            {"repository": "offline-python-wheels", "revision": "linux-cp311", "files": files}, indent=2) + "\n")
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if path.name == "requirements.lock.txt":
                continue
            if not path.is_file() or any(part in ("__pycache__", ".pytest_cache", "configs", "reports") for part in relative.parts):
                continue
            if path.suffix not in (".py", ".md", ".sh", ".txt", ".json"):
                continue
            content = path.read_bytes().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(content)
            info.mode = 0o700 if path.suffix == ".sh" else 0o600
            archive.addfile(info, io.BytesIO(content))
    print(json.dumps({"bundle": str(root / "source.tar.gz"), "sha256": sha256(root / "source.tar.gz")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "bundle", "receive"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--socket", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "download":
        download(arguments.root, arguments.credentials)
    elif arguments.action == "receive":
        receive(arguments.root, arguments.socket)
    else:
        bundle(arguments.root)