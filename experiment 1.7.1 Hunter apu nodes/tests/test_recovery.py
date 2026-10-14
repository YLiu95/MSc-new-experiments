import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ranker.data import write_json
from ranker.publish import git_blob_file
from ranking17 import publish, recovery
from ranking17.recovery import backup, backup_root, digest, verify


def make_checkpoint(directory, contents):
    directory.mkdir()
    payload = directory / "recovery.pt"
    payload.write_bytes(contents)
    write_json(directory / "manifest.json", {payload.name: {"bytes": len(contents), "sha256": digest(payload)}})
    (directory / "COMPLETE").write_text(digest(directory / "manifest.json") + "\n")


def test_private_copy_and_corruption_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ROOT", str(tmp_path / "private"))
    private = backup_root()
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    source = tmp_path / "source"
    make_checkpoint(source, b"recovery fixture")
    destination = private / "checkpoint"
    backup(source, destination)
    assert verify(source) == verify(destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    backup(source, destination)
    (destination / "recovery.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify(destination)


def test_existing_different_backup_is_rejected(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    make_checkpoint(source, b"first")
    make_checkpoint(destination, b"second")
    with pytest.raises(ValueError, match="differs from the source"):
        backup(source, destination)


@pytest.mark.parametrize("corrupt", [False, True])
def test_parallel_backup_verifies_every_shard_before_promotion(tmp_path, monkeypatch, corrupt):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    for shard in range(4):
        (source / f"recovery-{shard:02d}.pt").write_bytes(f"shard-{shard}".encode())
    write_json(source / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                                          for path in source.iterdir()})
    (source / "COMPLETE").write_text(digest(source / "manifest.json") + "\n")
    copy_barrier, hash_barrier = threading.Barrier(4), threading.Barrier(4)
    hash_threads = set()
    lock = threading.Lock()
    original_copy, original_digest = recovery.shutil.copy2, recovery.digest

    def checked_copy(source_path, destination_path):
        if source_path.name.startswith("recovery-"):
            copy_barrier.wait(timeout=10)
        result = original_copy(source_path, destination_path)
        if corrupt and source_path.name == "recovery-00.pt":
            destination_path.write_bytes(b"corrupt")
        return result

    def checked_digest(path):
        if path.name.startswith("recovery-") and threading.current_thread() is not threading.main_thread():
            with lock:
                hash_threads.add(threading.get_ident())
            hash_barrier.wait(timeout=10)
        return original_digest(path)

    monkeypatch.setattr(recovery.shutil, "copy2", checked_copy)
    monkeypatch.setattr(recovery, "digest", checked_digest)
    if corrupt:
        with pytest.raises(ValueError, match="digest mismatch"):
            backup(source, destination)
        assert not destination.exists()
    else:
        backup(source, destination)
        assert verify(source) == verify(destination)
        assert not destination.with_name(destination.name + ".staging").exists()
    assert len(hash_threads) >= 4


def test_best_upload_replaces_previous_step_and_verifies_files(tmp_path, monkeypatch):
    class FakeHub:
        files = {}

        def whoami(self):
            return {"name": "YL95"}

        def create_repo(self, *args, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(private=False, siblings=[SimpleNamespace(rfilename=name, lfs=None,
                blob_id=git_blob_file(path), size=path.stat().st_size) for name, path in self.files.items()])

        def create_commit(self, repo, operations, **kwargs):
            self.files = {operation.path_in_repo: tmp_path / f"uploaded-{operation.path_in_repo.replace('/', '-')}"
                          for operation in operations}
            for operation in operations:
                self.files[operation.path_in_repo].write_bytes(Path(operation.path_or_fileobj).read_bytes())
            return SimpleNamespace(oid="test-commit")

    api = FakeHub()
    monkeypatch.setattr(publish, "HfApi", lambda token: api)
    monkeypatch.setattr(publish, "credential", lambda name: "test-token")
    (tmp_path / "reports").mkdir()
    for step in (1, 5):
        checkpoint = tmp_path / "checkpoints" / f"step_{step:08d}"
        checkpoint.mkdir(parents=True)
        for shard in range(4):
            (checkpoint / f"weights-{shard:02d}.safetensors").write_bytes(f"{step}-{shard}".encode())
        write_json(checkpoint / "state.json", {"step": step})
        write_json(checkpoint / "config.json", {"experiment": "1.7.1", "tp": 4})
        write_json(checkpoint / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in checkpoint.iterdir()})
        (checkpoint / "COMPLETE").write_text(digest(checkpoint / "manifest.json") + "\n")
        write_json(tmp_path / "best.json", {"directory": str(checkpoint.relative_to(tmp_path))})
        publish.best(tmp_path)
        assert len(api.files) == 7
        assert api.files["best/weights-00.safetensors"].read_bytes() == f"{step}-0".encode()
        assert (tmp_path / "reports" / "best_publication.json").is_file()