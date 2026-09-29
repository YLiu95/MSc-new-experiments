from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import torch

from ranker.checkpoints import pointer
from ranker.contract import REVISION, price_to_returns
from ranker.data import MarketPanel, PREPARATION_POLICY, write_json, write_market
from ranker.evaluate import build_manifest
from ranker.model import ModelConfig
import ranker.train as training


def test_real_format_next_update_after_restore_matches_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MARKETS", ("AU",))
    monkeypatch.setattr(training, "ModelConfig", lambda tickers: ModelConfig(tickers=tickers, width=32, heads=4,
                         feedforward=64, temporal_blocks=1, cross_blocks=1, dropout=0.0,
                         identity_dropout=0.1, activation_checkpointing=False))
    panel_root = tmp_path / "panel"
    dates = np.arange("2018-01-01", "2024-01-01", dtype="datetime64[D]")
    daily = np.linspace(0.001, 0.006, 6)[:, None] * np.arange(len(dates))[None, :]
    prices = np.exp(daily + 0.01 * np.sin(np.arange(len(dates))[None, :] / 9))
    write_market(panel_root / "AU", price_to_returns(prices), dates, [f"S{index}" for index in range(6)], 2.0)
    np.save(panel_root / "AU" / "ids.npy", np.arange(6, dtype=np.int32))
    write_json(panel_root / "meta.json", {"revision": REVISION, "preparation_policy": PREPARATION_POLICY,
                                          "n_tickers": 6, "return_scale_pct": 2.0})
    write_json(panel_root / "vocabulary.json", [{"market": "AU", "ticker": f"S{index}"} for index in range(6)])
    registry = build_manifest(panel_root / "evaluation", [MarketPanel(panel_root, "AU")],
                              "val", 7331, ((64, 7, 3),), max_cutoffs=2)
    write_json(panel_root / "evaluation" / "val-primary.json", registry)
    write_json(panel_root / "evaluation" / "val-heldout.json", registry)
    stop_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    backup_by = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

    def arguments(root, max_steps, resume=None):
        return Namespace(panel=panel_root, root=root, arm="C", stop_at=stop_at, backup_by=backup_by,
                         max_steps=max_steps, global_batch=4, microbatch=4, workers=0, seed=1337,
                         resume=resume, stop_file=None)

    training.train(arguments(tmp_path / "uninterrupted", 2))
    training.train(arguments(tmp_path / "restored", 1))
    partial = pointer(tmp_path / "restored" / "C", "latest")
    training.train(arguments(tmp_path / "restored", 2, partial))
    direct = torch.load(pointer(tmp_path / "uninterrupted" / "C", "latest") / "recovery.pt",
                        map_location="cpu", weights_only=True)
    replay = torch.load(pointer(tmp_path / "restored" / "C", "latest") / "recovery.pt",
                        map_location="cpu", weights_only=True)
    for name in direct["model"]:
        torch.testing.assert_close(direct["model"][name], replay["model"][name], atol=1e-6, rtol=1e-5)
    assert direct["state"]["scheduled"] == replay["state"]["scheduled"] == 8
    assert direct["state"]["step"] == replay["state"]["step"] == 2
    assert json.loads((tmp_path / "restored" / "C" / "TRAINING_DONE.json").read_text())["step"] == 2