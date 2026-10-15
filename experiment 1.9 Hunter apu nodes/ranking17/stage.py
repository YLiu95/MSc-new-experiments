import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
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


def sftp_quote(path):
    value = str(path)
    if any(character in value for character in '\r\n\0"\\*?[]'):
        raise ValueError(f"Unsafe SFTP path: {value!r}")
    return f'"{value}"'


def sftp_batch(root, destination, inventory):
    remote_root = PurePosixPath(destination)
    if not remote_root.is_absolute() or ".." in remote_root.parts:
        raise ValueError("SFTP destination must be an absolute workspace path")
    commands = []
    count = 0
    total = 0
    skipped = 0
    remaining = 0
    for folder in ("wheels", "dataset", "warmstart"):
        local = root / folder
        manifest_path = local / "download.json"
        manifest = json.loads(manifest_path.read_text())
        if not manifest["files"]:
            raise ValueError(f"Empty transfer manifest: {folder}")
        remote = remote_root / folder
        commands.append(f"-mkdir {sftp_quote(remote)}")
        commands.append(f"put {sftp_quote(manifest_path.resolve().as_posix())} "
                        f"{sftp_quote(remote / 'download.pending.json')}")
        commands.append(f"-rm {sftp_quote(remote / 'download.json')}")
        directories = {remote}
        for name, expected in manifest["files"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"Unsafe manifest path: {name}")
            path = local / name
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Local file size mismatch: {path}")
            parent = remote
            for part in relative.parts[:-1]:
                parent /= part
                if parent not in directories:
                    commands.append(f"-mkdir {sftp_quote(parent)}")
                    directories.add(parent)
            count += 1
            total += expected["bytes"]
            remote_size = inventory.get(f"{folder}/{name}")
            if remote_size is not None and (remote_size < 0 or remote_size > expected["bytes"]):
                raise ValueError(f"Invalid remote size, repair required: {folder}/{name}")
            if remote_size == expected["bytes"]:
                skipped += 1
                continue
            operation = "put" if remote_size is None else "put -a"
            commands.append(f"{operation} {sftp_quote(path.resolve().as_posix())} "
                            f"{sftp_quote(remote / relative)}")
            remaining += expected["bytes"] - (remote_size or 0)
    commands.append(f"put {sftp_quote(Path(__file__).resolve().as_posix())} "
                    f"{sftp_quote(remote_root / 'stage-transfer.py')}")
    commands.append("bye")
    batch = root / "resume.sftp"
    batch.write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"batch": str(batch), "files": count, "total_bytes": total,
                      "skipped_files": skipped, "remaining_bytes": remaining}), flush=True)
    return batch


def verify_transfers(root):
    pending = []
    for folder in ("wheels", "dataset", "warmstart"):
        destination = root / folder
        manifest_path = destination / "download.pending.json"
        manifest = json.loads(manifest_path.read_text())
        if not manifest["files"]:
            raise ValueError(f"Empty transfer manifest: {folder}")
        for name, expected in manifest["files"].items():
            path = (destination / name).resolve()
            path.relative_to(destination.resolve())
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Transferred artifact verification failed: {folder}/{name}")
        pending.append(manifest_path)
        print(json.dumps({"verified": folder, "files": len(manifest["files"])}), flush=True)
    for manifest_path in pending:
        manifest_path.replace(manifest_path.with_name("download.json"))
    print(json.dumps({"transfer_complete": True}), flush=True)


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
    parser.add_argument("action", choices=("download", "bundle", "receive", "sftp-batch", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--destination")
    parser.add_argument("--inventory", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "download":
        download(arguments.root, arguments.credentials)
    elif arguments.action == "receive":
        receive(arguments.root, arguments.socket)
    elif arguments.action == "sftp-batch":
        if not arguments.destination or not arguments.inventory:
            parser.error("sftp-batch requires --destination and --inventory")
        sftp_batch(arguments.root, arguments.destination, json.loads(arguments.inventory.read_text()))
    elif arguments.action == "verify":
        verify_transfers(arguments.root)
    else:
        bundle(arguments.root)