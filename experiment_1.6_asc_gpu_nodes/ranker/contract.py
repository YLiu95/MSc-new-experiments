from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MARKETS = ("AU", "CA", "CH", "CN", "DE", "FR", "GB", "HK", "IN", "JP", "KR", "NL", "US")
DATASET = "YL95/new_experiment_1-data"
REVISION = "bcbbefdbe2313673895eb1a0d354747a9f1624fa"
LENGTHS = tuple(range(64, 513, 8))
HELD_OUT = frozenset({(128, 7, 16), (256, 21, 64), (512, 90, 128)})
PRIMARY = ((64, 1, 3), (64, 7, 16), (128, 21, 32), (256, 7, 64),
           (256, 63, 64), (384, 90, 96), (512, 7, 128), (512, 63, 128))


@dataclass(frozen=True)
class Task:
    length: int
    horizon: int
    basket: int

    def __post_init__(self) -> None:
        if self.length not in LENGTHS or not 1 <= self.horizon <= min(90, self.length - 1) or not 3 <= self.basket <= 128:
            raise ValueError(f"Invalid ranking task: {self}")

    @property
    def key(self) -> tuple[int, int, int]:
        return self.length, self.horizon, self.basket


def tasks(include_held_out: bool = False):
    for length in LENGTHS:
        for horizon in range(1, min(90, length - 1) + 1):
            for basket in range(3, 129):
                task = Task(length, horizon, basket)
                if include_held_out or task.key not in HELD_OUT:
                    yield task


def price_to_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    valid = np.isfinite(prices) & (prices > 0)
    result = np.full(prices.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        differences = 100.0 * np.diff(np.log(prices), axis=1)
    result[:, 1:] = np.where(valid[:, 1:] & valid[:, :-1], differences, np.nan)
    return result


def historical_eligibility(returns: np.ndarray, length: int) -> np.ndarray:
    valid = np.isfinite(returns)
    result = np.zeros(valid.shape, dtype=bool)
    if valid.shape[1] < length:
        return result
    for start in range(0, len(valid), 256):
        block = valid[start:start + 256]
        cumulative = np.pad(np.cumsum(block, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        result[start:start + len(block), length - 1:] = cumulative[:, length:] - cumulative[:, :-length] == length
    return result


def future_labels(prices: np.ndarray, returns: np.ndarray, cutoff: int, horizon: int) -> np.ndarray:
    if cutoff < 0 or cutoff + horizon >= prices.shape[1]:
        raise ValueError("Label endpoint falls outside the observed market calendar")
    observed = np.isfinite(returns[:, cutoff + 1:cutoff + horizon + 1]).all(axis=1)
    result = np.full(prices.shape[0], np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 100.0 * (np.log(prices[observed, cutoff + horizon].astype(np.float64))
                          - np.log(prices[observed, cutoff].astype(np.float64)))
    result[observed] = values.astype(np.float32)
    return result


def labels_from_returns(returns: np.ndarray, cutoff: int, horizon: int) -> np.ndarray:
    if cutoff < 0 or cutoff + horizon >= returns.shape[1]:
        raise ValueError("Label endpoint falls outside the observed market calendar")
    future = returns[:, cutoff + 1:cutoff + horizon + 1]
    observed = np.isfinite(future).all(axis=1)
    labels = np.full(len(returns), np.nan, dtype=np.float32)
    labels[observed] = future[observed].sum(axis=1, dtype=np.float64).astype(np.float32)
    return labels


def split_masks(dates: np.ndarray, horizon: int, train_end: str = "2018-12-31",
                val_end: str = "2022-12-31", embargo: int = 90) -> dict[str, np.ndarray]:
    train_boundary = np.searchsorted(dates, np.datetime64(train_end), side="right") - 1
    val_boundary = np.searchsorted(dates, np.datetime64(val_end), side="right") - 1
    if train_boundary < 0 or val_boundary < train_boundary or horizon < 1 or horizon > 90:
        raise ValueError("Invalid chronological split boundaries or horizon")
    positions = np.arange(len(dates))
    return {
        "train": positions <= train_boundary - horizon,
        "val": (positions >= train_boundary + embargo) & (positions <= val_boundary - horizon),
        "test": (positions >= val_boundary + embargo) & (positions + horizon < len(dates)),
    }