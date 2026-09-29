import json

import numpy as np
import pytest

from ranker.contract import Task, labels_from_returns, price_to_returns
from ranker.data import MarketPanel, write_market


def test_panel_uses_only_history_for_baskets(tmp_path):
    prices = np.exp(np.arange(150)[None, :] * np.array([0.01, -0.02, 0.01])[:, None])
    prices[2, 80] = np.nan
    returns = price_to_returns(prices)
    dates = np.arange("2018-01-01", 150, dtype="datetime64[D]")
    write_market(tmp_path / "AU", returns, dates, ["A", "B", "C"], 2.0)
    np.save(tmp_path / "AU" / "ids.npy", np.array([0, 1, -1], dtype=np.int32))
    panel = MarketPanel(tmp_path, "AU")
    np.testing.assert_array_equal(panel.eligible(75, 64), [0, 1, 2])
    sample = panel.sample(75, Task(64, 7, 3), np.array([0, 1, 2]), unknown_id=3)
    np.testing.assert_array_equal(sample["ticker_ids"], [0, 1, 3])
    np.testing.assert_array_equal(sample["observed"], [True, True, False])
    assert sample["inputs"].shape == (3, 64)
    assert sample["labels"][1] == pytest.approx(-14)
    assert json.loads((tmp_path / "AU" / "symbols.json").read_text()) == ["A", "B", "C"]
    with pytest.raises(ValueError):
        panel.sample(75, Task(64, 7, 3), np.array([0, 0, 2]), unknown_id=3)


def test_fp64_label_reduction_and_missing_future():
    returns = np.array([[np.nan, 1, -2, -3], [np.nan, 1, np.nan, 3]], dtype=np.float32)
    labels = labels_from_returns(returns, 1, 2)
    assert labels[0] == -5
    assert np.isnan(labels[1])