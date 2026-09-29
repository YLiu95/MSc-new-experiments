import stat

import pytest

from ranker.data import write_json
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