import json
from dataclasses import asdict
import sqlite3

import pytest
import torch

from ranker.checkpoints import save_checkpoint
from ranker.publish import GITHUB_FOLDER, git_blob, github_files, model_files
from ranker.data import PREPARATION_POLICY, write_json
from ranker.model import ModelConfig, RankingModel


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