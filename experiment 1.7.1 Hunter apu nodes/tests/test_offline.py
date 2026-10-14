import json

from ranker.data import write_json
from ranking17.offline import maintain
from ranking17.publish import public_snapshot
from ranking17.recovery import digest, verify


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