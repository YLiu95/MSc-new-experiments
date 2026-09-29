import math

import numpy as np
import pytest

from ranker.contract import Task, price_to_returns
from ranker.data import MarketPanel, write_market
from ranker.sampler import Sampler, coefficients, unrank


def test_exhaustive_bijection_and_large_integer_domains():
    for n in range(3, 10):
        for k in range(3, n + 1):
            size = math.comb(n, k)
            seen = {unrank(n, k, rank) for rank in range(size)}
            assert len(seen) == size
            for seed in range(3):
                multiplier, offset = coefficients(seed, n * 100 + k, size)
                assert len({(multiplier * index + offset) % size for index in range(size)}) == size
    size = math.comb(6861, 128)
    assert size > 2**63
    multiplier, offset = coefficients(1337, 12345, size)
    basket = unrank(6861, 128, (multiplier + offset) % size)
    assert len(basket) == 128 and len(set(basket)) == 128
    with pytest.raises(ValueError):
        unrank(6, 3, math.comb(6, 3))


def test_checkpoint_preserves_next_assignment_and_exhaustion(tmp_path):
    dates = np.arange("2018-01-01", "2018-06-01", dtype="datetime64[D]")
    prices = np.exp(np.arange(len(dates))[None, :] * np.linspace(0.001, 0.006, 6)[:, None])
    write_market(tmp_path / "panel" / "AU", price_to_returns(prices), dates,
                 [f"S{index}" for index in range(6)], 2.0)
    np.save(tmp_path / "panel" / "AU" / "ids.npy", np.arange(6, dtype=np.int32))
    panel = MarketPanel(tmp_path / "panel", "AU")
    task = Task(64, 1, 3)
    sampler = Sampler([panel], tmp_path / "initial.sqlite", fixed=task)
    first = sampler.issue(80)
    sampler.save(tmp_path / "snapshot.sqlite")
    second = sampler.issue(80)
    restored = Sampler.restore([panel], tmp_path / "snapshot.sqlite", tmp_path / "restored.sqlite", fixed=task)
    assert second == restored.issue(80)
    assert len({sample.identity for sample in first + second}) == 160
    assert sampler.issued == restored.issued == 160
    sampler.close()
    restored.close()