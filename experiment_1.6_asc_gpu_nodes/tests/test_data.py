import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ranker.contract import REVISION, Task, historical_eligibility, labels_from_returns, price_to_returns
from ranker.data import MarketPanel, mask_imputed_returns, prepare, source_imputation_mask, write_json, write_market


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


def test_source_imputation_masks_both_adjacent_returns_without_changing_earlier_basket():
    prices = np.exp(np.arange(90)[None, :] * np.array([0.01, 0.02, -0.01])[:, None])
    raw = price_to_returns(prices)
    flags = np.zeros(raw.shape, dtype=bool)
    flags[1, 75] = True
    cutoff = 74
    earlier = historical_eligibility(raw, 64)[:, cutoff]
    corrected, excluded = mask_imputed_returns(raw, flags)
    assert excluded == 2
    assert np.isnan(corrected[1, 75:77]).all()
    np.testing.assert_array_equal(historical_eligibility(corrected, 64)[:, cutoff], earlier)
    assert np.isnan(labels_from_returns(corrected, cutoff, 7)[1])
    assert not historical_eligibility(corrected, 64)[1, 76]


def test_pinned_parquet_flags_align_with_ticker_dates_and_training_split(tmp_path):
    path = tmp_path / "flags.parquet"
    table = pa.table({"ticker": ["B", "A", "A", "B"],
                      "date": pa.array(np.array(["2018-12-31", "2018-12-31", "2019-01-01", "2019-01-01"],
                                                dtype="datetime64[D]")),
                      "flag_imputed": [False, True, None, True]})
    pq.write_table(table, path)
    dates = np.array(["2018-12-31", "2019-01-01"], dtype="datetime64[D]")
    flags, audit = source_imputation_mask([path], dates, ["A", "B"])
    np.testing.assert_array_equal(flags, [[True, False], [False, True]])
    assert audit["source_imputed_prices_rejected"] == 2
    assert audit["source_imputed_train_prices"] == 1
    assert audit["source_imputed_validation_prices"] == 1


def test_existing_unmasked_panel_cannot_be_reused(tmp_path):
    write_json(tmp_path / "panel" / "meta.json", {"revision": REVISION, "source": "old"})
    with pytest.raises(ValueError, match="provenance"):
        prepare(tmp_path)