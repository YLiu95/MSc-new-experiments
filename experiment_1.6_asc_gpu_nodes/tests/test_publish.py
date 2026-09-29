import json

import pytest

from ranker.publish import GITHUB_FOLDER, git_blob, github_files, model_files


def test_private_source_allowlist_and_git_blob(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MODEL_CARD.md").write_text("private card\n")
    (source / "ranker").mkdir()
    (source / "ranker" / "model.py").write_text("print(1)\n")
    (source / "ranker" / "secret.env").write_text("no\n")
    (source / "raw.npy").write_bytes(b"private panel")
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n")
    files = github_files(source, plan)
    assert sorted(files) == [f"{GITHUB_FOLDER}/EXPERIMENT_PLAN.md", f"{GITHUB_FOLDER}/MODEL_CARD.md",
                             f"{GITHUB_FOLDER}/ranker/model.py"]
    assert git_blob(b"test") == "30d74d258442c7c65512eafab474568dd706c430"


def test_public_allowlist_rejects_incomplete_training(tmp_path):
    root = tmp_path / "artifacts"
    (root / "C").mkdir(parents=True)
    (root / "C" / "raw.npy").write_bytes(b"private")
    with pytest.raises(ValueError):
        model_files(root, "C", {"repository": "YLiu95/MSc-new-experiments", "private": True,
                                "branch": "main", "commit": "abc"})
    assert not (root / "publication" / "README.md").exists()