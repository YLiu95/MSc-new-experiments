import numpy as np
import pytest

from ranker.contract import HELD_OUT, LENGTHS, Task, future_labels, historical_eligibility, price_to_returns, split_masks, tasks


def test_task_space_and_holdouts():
    assert len(LENGTHS) == 57
    assert sum(min(90, length - 1) for length in LENGTHS) == 5070
    assert sum(1 for _ in tasks(True)) == 638820
    assert sum(1 for _ in tasks()) == 638817
    assert HELD_OUT.isdisjoint(task.key for task in tasks())
    for values in ((56, 1, 3), (513, 1, 3), (64, 64, 3), (96, 91, 3), (64, 1, 2), (64, 1, 129)):
        with pytest.raises(ValueError):
            Task(*values)


def test_labels_are_signed_and_future_availability_does_not_change_inputs():
    days = np.arange(520)
    prices = np.stack([np.exp(days * 0.01), np.exp(-days * 0.02), np.exp(days * 0.01)])
    returns = price_to_returns(prices)
    before = historical_eligibility(returns, 512)[:, 512]
    assert before.all()
    np.testing.assert_allclose(future_labels(prices, returns, 512, 7), [7, -14, 7], atol=1e-4)
    prices[2, 513] = np.nan
    returns = price_to_returns(prices)
    np.testing.assert_array_equal(historical_eligibility(returns, 512)[:, 512], before)
    labels = future_labels(prices, returns, 512, 7)
    assert labels[0] == pytest.approx(7)
    assert labels[1] == pytest.approx(-14)
    assert np.isnan(labels[2])
    assert not historical_eligibility(returns[:, :512], 512).any()


def test_session_split_endpoints_and_embargo():
    dates = np.arange("2018-01-01", "2024-01-01", dtype="datetime64[D]")
    masks = split_masks(dates, 90)
    train_end = np.searchsorted(dates, np.datetime64("2018-12-31"), side="right") - 1
    val_end = np.searchsorted(dates, np.datetime64("2022-12-31"), side="right") - 1
    assert np.flatnonzero(masks["train"])[-1] == train_end - 90
    assert np.flatnonzero(masks["val"])[0] == train_end + 90
    assert np.flatnonzero(masks["val"])[-1] == val_end - 90
    assert np.flatnonzero(masks["test"])[0] == val_end + 90
    assert np.flatnonzero(masks["test"])[-1] == len(dates) - 91