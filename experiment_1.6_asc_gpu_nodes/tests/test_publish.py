import json
from dataclasses import asdict
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from ranker.checkpoints import pointer, save_checkpoint
from ranker.publish import GITHUB_FOLDER, git_blob, github_files, model_files
from ranker.data import PREPARATION_POLICY, write_json
from ranker.model import ModelConfig, RankingModel
from scripts.publish_arms import REPOSITORIES, files_for_arm, publish_arm


class Ledger:
    def save(self, path):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE progress (id INTEGER PRIMARY KEY, issued INTEGER)")
            connection.execute("INSERT INTO progress VALUES (1, 320)")


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


def test_public_file_set_is_one_verifiable_best_and_latest_without_private_panels(tmp_path):
    root = tmp_path / "artifacts"
    selected = root / "C"
    config = ModelConfig(tickers=6, width=32, heads=4, feedforward=64,
                         temporal_blocks=1, cross_blocks=1, dropout=0)
    model = RankingModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(selected, model, optimizer, {"step": 1, "scheduled": 320, "world": 1},
                    Ledger(), {"model": asdict(config)}, True)
    save_checkpoint(selected, model, optimizer, {"step": 2, "scheduled": 640, "world": 1},
                    Ledger(), {"model": asdict(config)}, False)
    write_json(selected / "TRAINING_DONE.json", {"step": 2})
    write_json(selected / "reports" / "training_summary.json", {"best_step": 1})
    (selected / "runs").mkdir()
    (selected / "runs" / "events.out.tfevents.test").write_text("aggregate only")
    (root / "panel").mkdir()
    write_json(root / "panel" / "meta.json", {"return_scale_pct": 3.4, "input_clip": 8,
                                                 "n_tickers": 6, "preparation_policy": PREPARATION_POLICY})
    write_json(root / "panel" / "vocabulary.json", [{"market": "AU", "ticker": "A"}])
    (root / "panel" / "raw.npy").write_bytes(b"private")
    for name in ("baselines.json", "evaluation_registry.json"):
        write_json(root / "reports" / name, {})
    source = {"repository": "YLiu95/MSc-new-experiments", "private": True,
              "branch": "main", "commit": "abc123"}
    files = model_files(root, "C", source)
    assert {"best/weights.safetensors", "best/inference_manifest.json", "latest/recovery.pt",
            "latest/sampler.sqlite", "latest/manifest.json", "latest/COMPLETE",
            "runs/events.out.tfevents.test"}.issubset(files)
    assert all("raw.npy" not in name and ".env" not in name for name in files)
    assert files["README.md"].read_text() == ("[Private model card](https://github.com/"
        "YLiu95/MSc-new-experiments/blob/abc123/experiment_1.6_asc_gpu_nodes/MODEL_CARD.md)\n")


def three_repo_fixture(tmp_path, arm, best_step, latest_step):
    root = tmp_path / "artifacts"
    selected = root / arm
    config = ModelConfig(tickers=6, width=32, heads=4, feedforward=64,
                         temporal_blocks=1, cross_blocks=1, dropout=0)
    model = RankingModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    saved_config = {"model": asdict(config), "arm": arm,
                    "data": {"preparation_policy": PREPARATION_POLICY},
                    "validation": {"primary": "frozen-hash"}}
    write_json(selected / "config.json", saved_config)
    save_checkpoint(selected, model, optimizer,
                    {"step": best_step, "scheduled": best_step * 320, "world": 1},
                    Ledger(), saved_config, True)
    if latest_step != best_step:
        save_checkpoint(selected, model, optimizer,
                        {"step": latest_step, "scheduled": latest_step * 320, "world": 1},
                        Ledger(), saved_config, False)
    write_json(selected / "TRAINING_DONE.json", {"step": latest_step})
    write_json(selected / "reports" / "training_summary.json", {"full_validation_selection": True})
    (selected / "validation.jsonl").write_text("{}\n")
    (selected / "history.jsonl").write_text("{}\n")
    (selected / "runs").mkdir()
    (selected / "runs" / "events.out.tfevents.test").write_text("aggregate only")
    write_json(root / "panel" / "meta.json", {"revision": "bcbbefdbe2313673895eb1a0d354747a9f1624fa",
                                                   "preparation_policy": PREPARATION_POLICY,
                                                   "return_scale_pct": 3.4029, "input_clip": 8, "n_tickers": 6})
    write_json(root / "panel" / "vocabulary.json", [{"market": "AU", "ticker": "A"}])
    write_json(root / "panel" / "evaluation" / "val-primary.json", {"sha256": "frozen-hash"})
    (root / "panel" / "raw.npy").write_bytes(b"private panel")
    for name in ("baselines.json", "evaluation_registry.json"):
        write_json(root / "reports" / name, {})
    write_json(root / "control" / "publication_authorization.json",
               {"confirmed_by_user": True, "training_root": str(root.resolve())})
    source = {"repository": "YLiu95/MSc-new-experiments", "private": True,
              "branch": "main", "commit": "abc123"}
    return root, source


def test_three_public_repositories_skip_duplicate_latest(tmp_path):
    assert set(REPOSITORIES) == {"A", "B", "C"}
    assert REPOSITORIES["C"] == "YL95/experiment-1.6-asc-gpu-nodes"
    assert len(set(REPOSITORIES.values())) == 3
    root, source = three_repo_fixture(tmp_path, "B", 1, 1)
    paths, same = files_for_arm(root, "B", source)
    assert same and not any(name.startswith("latest/") for name in paths)
    assert {"best/weights.safetensors", "best/inference_manifest.json", "reports/validation.jsonl",
            "reports/history.jsonl", "runs/events.out.tfevents.test"}.issubset(paths)
    assert paths["README.md"].read_text().startswith("[Private model card](https://")
    assert not any("raw.npy" in name for name in paths)
    assert json.loads(paths["reconstruction.json"].read_text())["public_recovery_included"] is False
    (root / "control" / "publication_authorization.json").unlink()
    with pytest.raises(FileNotFoundError):
        files_for_arm(root, "B", source)


def test_distinct_latest_is_complete_even_when_best_remains_initial(tmp_path):
    root, source = three_repo_fixture(tmp_path, "A", 0, 1)
    paths, same = files_for_arm(root, "A", source)
    assert not same
    assert {"latest/recovery.pt", "latest/sampler.sqlite", "latest/manifest.json",
            "latest/COMPLETE", "latest/rng-0000.pt"}.issubset(paths)
    assert json.loads(paths["reconstruction.json"].read_text())["best_at_initialization"] is True


def test_failed_public_commit_preserves_private_recovery_without_false_verification(tmp_path):
    root, source = three_repo_fixture(tmp_path, "A", 1, 2)

    class InterruptedApi:
        def create_repo(self, **arguments):
            return None

        def model_info(self, repository):
            return SimpleNamespace(private=False)

        def create_commit(self, **arguments):
            raise RuntimeError("upload interrupted")

    with pytest.raises(RuntimeError, match="upload interrupted"):
        publish_arm(root, "A", source, InterruptedApi())
    assert pointer(root / "A", "best").is_dir()
    assert pointer(root / "A", "latest").is_dir()
    assert not (root / "reports" / "public_backup_verification_A.json").exists()