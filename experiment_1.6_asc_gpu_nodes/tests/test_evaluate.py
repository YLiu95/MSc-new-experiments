import numpy as np
import pytest
import torch

from ranker.contract import Task, price_to_returns
from ranker.data import MarketPanel, write_json, write_market
from ranker.evaluate import average_ranks, basket_metrics, build_manifest, evaluate_model, macro_spearman
from ranker.model import ModelConfig, RankingModel


def test_tied_ranks_prediction_ties_and_unobserved_selections():
    np.testing.assert_array_equal(average_ranks(np.array([3, 1, 3, 2])), [3.5, 1, 3.5, 2])
    result = basket_metrics(np.array([2., 2., 1., 0.]), np.array([3., 1., -2., np.nan]))
    assert result["spearman"] > 0.8
    assert result["pairs"] == 3 and result["correct_pairs"] == 2.5
    assert result["top_observed"] == 1 and result["bottom_observed"] == 0
    assert basket_metrics(np.array([1., 1., 1.]), np.array([1., 2., 3.]))["spearman"] == 0
    assert basket_metrics(np.array([1., 2., 3.]), np.array([2., 2., 2.]))["spearman"] is None


def test_macro_does_not_silently_drop_an_entire_task():
    rows = [{"spearman": 0.5, "task": (64, 1, 3), "market": "AU", "cutoff": "2019-06-01",
             "observed": 3, "unobserved": 0, "pairs": 3, "correct_pairs": 2.,
             "top_sum": 1., "top_observed": 1, "bottom_sum": -1., "bottom_observed": 1}]
    assert macro_spearman(rows, ((64, 1, 3),))["macro_spearman"] == 0.5
    with pytest.raises(ValueError):
        macro_spearman(rows, ((64, 1, 3), (64, 7, 16)))


def test_manifest_is_frozen_and_does_not_depend_on_future_labels(tmp_path):
    dates = np.arange("2018-01-01", "2024-01-01", dtype="datetime64[D]")
    prices = np.exp(np.arange(len(dates))[None, :] * np.linspace(0.001, 0.006, 6)[:, None])
    returns = price_to_returns(prices)
    write_market(tmp_path / "AU", returns, dates, [f"S{index}" for index in range(6)], 2.0)
    np.save(tmp_path / "AU" / "ids.npy", np.arange(6, dtype=np.int32))
    panel = MarketPanel(tmp_path, "AU")
    first = build_manifest(tmp_path / "eval", [panel], "val", 7331, ((64, 7, 3),), 3)
    assert first["baskets"] == 12
    assert first == build_manifest(tmp_path / "eval", [panel], "val", 7331, ((64, 7, 3),), 3)
    write_json(tmp_path / "meta.json", {"n_tickers": 6})
    model = RankingModel(ModelConfig(tickers=6, width=32, heads=4, feedforward=64,
                                     temporal_blocks=1, cross_blocks=1, dropout=0)).eval()
    report = evaluate_model(model, [panel], tmp_path, tmp_path / "eval", first, torch.device("cpu"))
    assert report["model"]["baskets"] == 12
    assert report["momentum"]["macro_spearman"] > 0
    first["monitor_line_indices"] = [0, 4]
    monitor = evaluate_model(model, [panel], tmp_path, tmp_path / "eval", first, torch.device("cpu"), monitoring=True)
    assert monitor["model"]["baskets"] == 2
    with pytest.raises(ValueError):
        build_manifest(tmp_path / "eval", [panel], "val", 7332, ((64, 7, 3),), 3)