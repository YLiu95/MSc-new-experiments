import hashlib
import json

import pytest

from ranking17.stage import sftp_batch, sftp_quote, verify_transfers


def artifacts(root, manifest_name):
    for folder in ("wheels", "dataset", "warmstart"):
        destination = root / folder
        (destination / "nested").mkdir(parents=True)
        (destination / "nested" / "file.bin").write_bytes(b"verified")
        manifest = {"files": {"nested/file.bin": {
            "bytes": 8, "sha256": hashlib.sha256(b"verified").hexdigest()}}}
        (destination / manifest_name).write_text(json.dumps(manifest))


def test_sftp_batch_resumes_files_and_defers_completion(tmp_path):
    artifacts(tmp_path, "download.json")
    inventory = {"wheels/nested/file.bin": 8, "dataset/nested/file.bin": 3}
    batch = sftp_batch(tmp_path, "/workspace/experiment", inventory)
    commands = batch.read_text().splitlines()
    assert len([line for line in commands if line.startswith("put -a ")]) == 1
    assert any(line.startswith("put ") and line.endswith('/warmstart/nested/file.bin"') for line in commands)
    assert not any(line.startswith("put ") and line.endswith('/wheels/nested/file.bin"') for line in commands)
    assert len([line for line in commands if "download.pending.json" in line]) == 3
    assert not any(line.startswith("put ") and line.endswith('/download.json"') for line in commands)
    assert commands[-1] == "bye"


def test_sftp_rejects_oversized_remote_file(tmp_path):
    artifacts(tmp_path, "download.json")
    with pytest.raises(ValueError, match="repair required"):
        sftp_batch(tmp_path, "/workspace/experiment", {"wheels/nested/file.bin": 9})


@pytest.mark.parametrize("content", [b"corrupt!", b"short", b"oversized content"])
def test_verification_rejects_corruption_before_any_completion(tmp_path, content):
    artifacts(tmp_path, "download.pending.json")
    (tmp_path / "warmstart" / "nested" / "file.bin").write_bytes(content)
    with pytest.raises(ValueError, match="warmstart/nested/file.bin"):
        verify_transfers(tmp_path)
    assert not list(tmp_path.glob("*/download.json"))
    (tmp_path / "warmstart" / "nested" / "file.bin").write_bytes(b"verified")
    verify_transfers(tmp_path)
    assert len(list(tmp_path.glob("*/download.json"))) == 3


@pytest.mark.parametrize("path", ['bad"name', "bad\nname", "bad*name", "bad\\name"])
def test_sftp_rejects_command_injection_and_globs(path):
    with pytest.raises(ValueError):
        sftp_quote(path)