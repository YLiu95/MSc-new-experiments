from dataclasses import asdict
import sqlite3

import pytest
import torch

from ranker.checkpoints import pointer, restore_checkpoint, save_checkpoint, verify
from ranker.model import ModelConfig, RankingModel


class FixtureLedger:
    def save(self, path):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE progress (id INTEGER PRIMARY KEY, issued INTEGER)")
            connection.execute("INSERT INTO progress VALUES (1, 4)")


def test_best_and_latest_recover_optimizer_and_reject_corruption(tmp_path):
    model_config = ModelConfig(tickers=6, width=32, heads=4, feedforward=64,
                               temporal_blocks=1, cross_blocks=1, dropout=0)
    torch.manual_seed(1337)
    model = RankingModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(1, 3, 64)
    ids = torch.tensor([[0, 1, 2]])
    market = torch.tensor([0])
    task = torch.tensor([[64, 7, 3]])
    model(inputs, ids, market, task)[0].square().mean().backward()
    optimizer.step()
    config = {"model": asdict(model_config), "data": {"revision": "pinned"}}
    state = {"step": 1, "scheduled": 4, "world": 1}
    directory = save_checkpoint(tmp_path, model, optimizer, state, FixtureLedger(), config, True)
    assert directory == pointer(tmp_path, "latest") == pointer(tmp_path, "best")
    assert verify(directory)["is_best"]
    replacement = RankingModel(model_config)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3)
    restored, snapshot = restore_checkpoint(directory, replacement, replacement_optimizer, config)
    assert restored == state and snapshot.is_file()
    assert len(replacement_optimizer.state) == len(optimizer.state)
    model.eval()
    replacement.eval()
    torch.testing.assert_close(model(inputs, ids, market, task)[0], replacement(inputs, ids, market, task)[0])
    with pytest.raises(ValueError):
        restore_checkpoint(directory, replacement, replacement_optimizer, {"model": {}, "data": {}})
    (directory / "state.json").write_text("damaged")
    with pytest.raises(ValueError):
        pointer(tmp_path, "latest")