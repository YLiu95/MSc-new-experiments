from dataclasses import asdict

import sqlite3
import torch
import pytest

from ranker.backup import local
from ranker.checkpoints import save_checkpoint
from ranker.model import ModelConfig, RankingModel


class Ledger:
    def save(self, path):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE progress (id INTEGER PRIMARY KEY, issued INTEGER)")
            connection.execute("INSERT INTO progress VALUES (1, 4)")


def test_backup_copies_only_selected_complete_states_and_rechecks_digests(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    root = tmp_path / "C"
    config = ModelConfig(tickers=5, width=32, heads=4, feedforward=64, temporal_blocks=1,
                         cross_blocks=1, dropout=0)
    model = RankingModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    state = {"step": 0, "scheduled": 4, "world": 1}
    save_checkpoint(root, model, optimizer, state, Ledger(), {"model": asdict(config)}, True)
    (root / "raw.npy").write_bytes(b"private panel")
    manifest = local(root, "12345")
    assert "best/weights.safetensors" in manifest["files"]
    assert "latest/recovery.pt" in manifest["files"]
    assert "raw.npy" not in manifest["files"]
    assert local(root, "12345") == manifest
    with pytest.raises(RuntimeError):
        local(root, "12345", backup_by="2000-01-01T00:00:00+00:00")