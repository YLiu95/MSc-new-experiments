import json
import shutil
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelCard

from ranker.data import write_json
from ranking17.offline import maintain
from ranking17.publish import SOURCE, cache_public_file, public_snapshot
from ranking17.recovery import digest, verify


def test_model_card_metadata_parses_for_publication():
    card = ModelCard.load(SOURCE / "MODEL_CARD.md")
    assert card.data.library_name == "pytorch"
    assert "tensorboard" in card.data.tags


def checkpoint(root, step, best=False):
    destination = root / "checkpoints" / f"step_{step:08d}"
    destination.mkdir(parents=True)
    write_json(destination / "config.json", {"tp": 4})
    write_json(destination / "state.json", {"step": step})
    for shard in range(4):
        (destination / f"weights-{shard:02d}.safetensors").write_bytes(f"{step}:{shard}".encode())
        (destination / f"recovery-{shard:02d}.pt").write_bytes(b"private optimizer")
    (destination / "sampler.sqlite").write_bytes(b"private sampler")
    write_json(destination / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                                               for path in destination.iterdir()})
    (destination / "COMPLETE").write_text(digest(destination / "manifest.json") + "\n")
    pointer = {"directory": destination.relative_to(root).as_posix()}
    write_json(root / "latest.json", pointer)
    if best:
        write_json(root / "best.json", pointer)
    return destination


def reseal(directory):
    write_json(directory / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                                             for path in directory.iterdir()
                                             if path.name not in ("manifest.json", "COMPLETE")})
    (directory / "COMPLETE").write_text(digest(directory / "manifest.json") + "\n")


@pytest.mark.parametrize("finished", [False, True])
def test_migration_retirement_preserves_replacement_and_parent_best(tmp_path, monkeypatch, finished):
    parent_root = tmp_path / "artifacts" / "job-285983.hunter-pbs01"
    root = tmp_path / "artifacts" / "job-six"
    backups = tmp_path / "backups"
    monkeypatch.setenv("BACKUP_ROOT", str(backups))
    checkpoint(parent_root, 180, best=True)
    maintain(parent_root)
    parent = checkpoint(parent_root, 185)
    write_json(parent / "state.json", {"step": 185, "scheduled": 10240, "world": 8})
    reseal(parent)
    maintain(parent_root)
    shutil.rmtree(backups / parent_root.name)
    if finished:
        write_json(parent_root / "TRAINING_DONE.json", {"step": 185, "scheduled": 10240, "world": 8})
    latest = checkpoint(root, 185)
    write_json(latest / "config.json", {"tp": 4, "migration": {
        "parent_checkpoint": str(parent), "parent_step": 185, "parent_scheduled": 10240,
        "parent_manifest_sha256": digest(parent / "manifest.json")}})
    write_json(latest / "state.json", {"step": 185, "scheduled": 10240, "world": 24})
    reseal(latest)
    if not finished:
        with pytest.raises(FileNotFoundError):
            maintain(root)
        assert parent.exists()
        return
    maintain(root)
    assert not parent.exists()
    assert not (parent_root / "exports").exists()
    assert verify(latest) == verify(backups / root.name / latest.name)
    best = root / "exports" / json.loads((root / "exports" / "best.json").read_text())["directory"]
    assert (best / "weights-00.safetensors").read_bytes() == b"180:0"
    assert (parent_root / "reports" / "migration-source" / "manifest.json").exists()
    assert (root / "reports" / "migration_retired.json").exists()
    maintain(root)
    assert best.exists()


def test_retention_preserves_best_export_and_verified_latest_recovery(tmp_path, monkeypatch):
    root = tmp_path / "run"
    backup = tmp_path / "backup"
    monkeypatch.setenv("BACKUP_ROOT", str(backup))
    first = checkpoint(root, 1, best=True)
    maintain(root)
    second = checkpoint(root, 2)
    maintain(root)
    assert not first.exists()
    assert second.exists()
    assert not (backup / root.name / first.name).exists()
    assert verify(backup / root.name / second.name) == verify(second)
    best = root / "exports" / json.loads((root / "exports" / "best.json").read_text())["directory"]
    assert (best / "weights-00.safetensors").read_bytes() == b"1:0"
    assert not list(best.glob("recovery*"))
    assert not (best / "sampler.sqlite").exists()


def test_public_snapshot_never_exposes_recovery_or_sampler(tmp_path, monkeypatch):
    root = tmp_path / "artifacts" / "job-test"
    monkeypatch.setenv("BACKUP_ROOT", str(tmp_path / "backups"))
    checkpoint(root, 1, best=True)
    maintain(root)
    write_json(root / "reports" / "preflight.json", {"passed": True})
    snapshot = public_snapshot(tmp_path)
    assert snapshot["job"] == "job-test"
    assert len([name for name in snapshot["files"] if name.endswith(".safetensors")]) == 8
    assert all("recovery" not in entry["path"] and "sampler" not in entry["path"]
               for entry in snapshot["files"].values())


@pytest.mark.parametrize("enough_space,valid_digest", [(True, True), (False, True), (True, False)])
def test_publication_cache_replacement_respects_disk_and_integrity(tmp_path, monkeypatch, enough_space, valid_digest):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"outdated cache")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(b"interrupted transfer")
    payload = b"verified replacement"
    import hashlib
    entry = {"path": "exports/weights.safetensors", "bytes": len(payload),
             "sha256": hashlib.sha256(payload if valid_digest else b"different payload").hexdigest()}

    def disk_usage(root):
        assert root == tmp_path
        assert not path.exists() and not temporary.exists()
        return SimpleNamespace(free=2 * 1024**3 + len(payload) - int(not enough_space))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, block_size):
            yield payload

    def get(url, **arguments):
        assert enough_space
        assert url == "http://localhost/files/exports/weights.safetensors"
        return Response()

    monkeypatch.setattr("ranking17.publish.shutil.disk_usage", disk_usage)
    monkeypatch.setattr("ranking17.publish.requests.get", get)
    if not enough_space or not valid_digest:
        with pytest.raises(RuntimeError if not enough_space else ValueError):
            cache_public_file(tmp_path, path.name, entry, "http://localhost")
        assert not path.exists()
    else:
        assert cache_public_file(tmp_path, path.name, entry, "http://localhost") == path
        assert path.read_bytes() == payload
        assert not temporary.exists()
        assert cache_public_file(tmp_path, path.name, entry, "http://localhost") == path