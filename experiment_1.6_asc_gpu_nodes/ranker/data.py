from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .contract import DATASET, LENGTHS, MARKETS, REVISION, Task, labels_from_returns, price_to_returns, split_masks


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def read_market(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    symbols = set()
    calendars = []
    observations = 0
    for path in paths:
        table = pq.read_table(path, columns=["ticker", "date"])
        symbols.update(pc.unique(table["ticker"]).to_pylist())
        dates = table["date"].to_numpy(zero_copy_only=False)
        if not np.array_equal(dates.astype("datetime64[ns]"), dates.astype("datetime64[D]").astype("datetime64[ns]")):
            raise ValueError(f"Non-session timestamps in {path.name}; audit required")
        calendars.append(np.unique(dates.astype("datetime64[D]")))
        observations += len(table)
    if not paths:
        raise ValueError("No market shards")
    names = sorted(symbols)
    dates = np.unique(np.concatenate(calendars))
    prices = np.full((len(names), len(dates)), np.nan, dtype=np.float64)
    seen = np.zeros(prices.shape, dtype=bool)
    invalid_prices = 0
    for path in paths:
        table = pq.read_table(path, columns=["ticker", "date", "adj_close_clean"])
        ticker_index = pd.Categorical(table["ticker"].to_pandas(), categories=names).codes.astype(np.int64)
        date_index = np.searchsorted(dates, table["date"].to_numpy(zero_copy_only=False).astype("datetime64[D]"))
        if (ticker_index < 0).any() or (date_index >= len(dates)).any():
            raise ValueError(f"Unaligned ticker/session in {path.name}; audit required")
        linear = ticker_index * len(dates) + date_index
        if len(np.unique(linear)) != len(linear) or seen.ravel()[linear].any():
            raise ValueError(f"Duplicate ticker/session in {path.name}; audit required")
        seen.ravel()[linear] = True
        values = table["adj_close_clean"].to_numpy(zero_copy_only=False).astype(np.float64)
        invalid = ~np.isfinite(values) | (values <= 0)
        invalid_prices += int(invalid.sum())
        values[invalid] = np.nan
        prices[ticker_index, date_index] = values
    return price_to_returns(prices), dates, names, {
        "observations": observations, "invalid_prices_rejected": invalid_prices, "duplicates_rejected": 0,
    }


def source_market(source: Path | None, market: str, snapshot: Path | None) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    if source is not None:
        directory = source / market
        info = json.loads((directory / "raw_info.json").read_text())
        return (np.load(directory / "raw.npy", mmap_mode="r", allow_pickle=False),
                np.load(directory / "dates.npy", allow_pickle=False),
                json.loads((directory / "symbols.json").read_text()),
                {"observations": info["observations"], "invalid_prices_rejected": info["invalid_prices"],
                 "duplicates_rejected": info["duplicates"]})
    return read_market(sorted((snapshot / "data" / market).glob("*.parquet")))


def audit_calendar(dates: np.ndarray) -> dict:
    if len(dates) < 2 or not np.all(dates[1:] > dates[:-1]):
        raise ValueError("Missing, duplicate, or out-of-order market sessions")
    business_day_gaps = np.busday_count(dates[:-1], dates[1:]) - 1
    suspicious = np.flatnonzero(business_day_gaps >= 10)
    return {"first": str(dates[0]), "last": str(dates[-1]), "sessions": len(dates),
            "weekday_gaps": int(np.count_nonzero(business_day_gaps)),
            "long_weekday_gaps": [{"after": str(dates[index]), "before": str(dates[index + 1]),
                                   "missing_weekdays": int(business_day_gaps[index])} for index in suspicious[:20]],
            "calendar_verified_by_exchange": False}


def write_market(directory: Path, raw: np.ndarray, dates: np.ndarray, symbols: list[str], scale: float) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if raw.shape != (len(symbols), len(dates)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Market panel or training scale is invalid")
    n_tickers, n_dates = raw.shape
    raw_output = np.lib.format.open_memmap(directory / "raw.tmp.npy", mode="w+", dtype=np.float32, shape=raw.shape)
    inputs = np.lib.format.open_memmap(directory / "inputs.tmp.npy", mode="w+", dtype=np.float16, shape=raw.shape)
    streaks = np.lib.format.open_memmap(directory / "streaks.tmp.npy", mode="w+", dtype=np.uint16, shape=(n_dates, n_tickers))
    counts = np.lib.format.open_memmap(directory / "counts.tmp.npy", mode="w+", dtype=np.uint16, shape=(len(LENGTHS), n_dates))
    for start in range(0, n_tickers, 128):
        block = np.asarray(raw[start:start + 128], dtype=np.float32)
        raw_output[start:start + len(block)] = block
        inputs[start:start + len(block)] = np.nan_to_num(np.clip(block / scale, -8, 8), nan=0).astype(np.float16)
    consecutive = np.zeros(n_tickers, dtype=np.uint16)
    for cutoff in range(n_dates):
        consecutive = (consecutive + 1) * np.isfinite(raw_output[:, cutoff])
        streaks[cutoff] = consecutive
    for index, length in enumerate(LENGTHS):
        counts[index] = np.count_nonzero(streaks >= length, axis=1)
    for array in (raw_output, inputs, streaks, counts):
        array.flush()
    del raw_output, inputs, streaks, counts
    for name in ("raw", "inputs", "streaks", "counts"):
        (directory / f"{name}.tmp.npy").replace(directory / f"{name}.npy")
    save_array(directory / "dates.npy", dates)
    write_json(directory / "symbols.json", symbols)
    return audit_calendar(dates)


def fit_scale(markets: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, dict]:
    total = 0.0
    squares = 0.0
    count = 0
    for raw, dates in markets:
        endpoint = np.searchsorted(dates, np.datetime64("2018-12-31"), side="right")
        for start in range(0, len(raw), 128):
            block = np.asarray(raw[start:start + 128, :endpoint], dtype=np.float64)
            values = block[np.isfinite(block)]
            total += float(values.sum(dtype=np.float64))
            squares += float(np.square(values).sum(dtype=np.float64))
            count += len(values)
    if not count:
        raise ValueError("Training period contains no finite returns")
    mean = total / count
    scale = float(np.sqrt(squares / count - mean * mean))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training-only standard deviation")
    return scale, {"count": count, "sum": total, "sum_squares": squares, "ddof": 0, "mean_subtracted": False}


def prepare(root: Path, source: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    panel_root = root / "panel"
    marker = panel_root / "meta.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta["revision"] != REVISION or (source is not None and meta["source"] != str(source.resolve())):
            raise ValueError("Prepared panel provenance differs from the requested revision/source")
        return panel_root
    snapshot = None
    if source is not None:
        source = source.resolve()
        meta = json.loads((source / "meta.json").read_text())
        if (meta["dataset"], meta["revision"]) != (DATASET, REVISION):
            raise ValueError("Cached source is not the pinned dataset revision")
    else:
        from dotenv import load_dotenv
        from huggingface_hub import snapshot_download
        load_dotenv(Path.home() / ".env", override=False)
        snapshot = Path(snapshot_download(repo_id=DATASET, repo_type="dataset", revision=REVISION,
                                         allow_patterns=[f"data/{market}/*.parquet" for market in MARKETS],
                                         token=os.environ.get("HF_TOKEN"), cache_dir=root / "cache" / "hf"))
    panel_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw_markets = []
    for market in MARKETS:
        raw, dates, symbols, audit = source_market(source, market, snapshot)
        raw_markets.append((raw, dates, symbols, audit))
    scale, statistics = fit_scale([(raw, dates) for raw, dates, _, _ in raw_markets])
    vocabulary = []
    report = []
    for market, (raw, dates, symbols, audit) in zip(MARKETS, raw_markets):
        directory = panel_root / market
        calendar = write_market(directory, raw, dates, symbols, scale)
        streaks = np.load(directory / "streaks.npy", mmap_mode="r")
        counts = np.load(directory / "counts.npy", mmap_mode="r")
        train_end = np.searchsorted(dates, np.datetime64("2018-12-31"), side="right") - 1
        eligible = (streaks[:train_end, :] >= 64) & (counts[0, :train_end, None] >= 3)
        seen = eligible.any(axis=0)
        ids = np.full(len(symbols), -1, dtype=np.int32)
        for local_id in np.flatnonzero(seen):
            ids[local_id] = len(vocabulary)
            vocabulary.append({"market": market, "ticker": symbols[local_id]})
        save_array(directory / "ids.npy", ids)
        info = {"market": market, "tickers": len(symbols), "training_identities": int(seen.sum()),
                "source": audit, "calendar": calendar,
                "history_eligible_cutoffs_by_length": [int(np.count_nonzero(counts[index, :train_end] >= 3)) for index in range(len(LENGTHS))]}
        write_json(directory / "info.json", info)
        report.append(info)
        print(json.dumps({"event": "market_prepared", "market": market, "scale": scale, "tickers": len(symbols)}), flush=True)
    write_json(panel_root / "vocabulary.json", vocabulary)
    metadata = {"dataset": DATASET, "revision": REVISION, "source": str(source) if source is not None else str(snapshot),
                "return_scale_pct": scale, "scale_statistics": statistics, "n_tickers": len(vocabulary),
                "markets": report, "train_end": "2018-12-31", "val_end": "2022-12-31", "embargo": 90,
                "input_clip": 8.0, "raw_dtype": "float32", "input_dtype": "float16"}
    write_json(root / "reports" / "data_audit.json", metadata)
    write_json(marker, metadata)
    return panel_root


class MarketPanel:
    def __init__(self, root: Path, market: str):
        directory = Path(root) / market
        self.market = market
        self.raw = np.load(directory / "raw.npy", mmap_mode="r", allow_pickle=False)
        self.inputs = np.load(directory / "inputs.npy", mmap_mode="r", allow_pickle=False)
        self.streaks = np.load(directory / "streaks.npy", mmap_mode="r", allow_pickle=False)
        self.counts = np.load(directory / "counts.npy", mmap_mode="r", allow_pickle=False)
        self.dates = np.load(directory / "dates.npy", mmap_mode="r", allow_pickle=False)
        self.ids = np.load(directory / "ids.npy", mmap_mode="r", allow_pickle=False)

    @lru_cache(maxsize=512)
    def cutoffs(self, task: Task, split: str) -> np.ndarray:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        mask = split_masks(self.dates, task.horizon)[split]
        return np.flatnonzero(mask & (self.counts[LENGTHS.index(task.length)] >= task.basket))

    def eligible(self, cutoff: int, length: int) -> np.ndarray:
        return np.flatnonzero(self.streaks[cutoff] >= length)

    def sample(self, cutoff: int, task: Task, tickers: np.ndarray, unknown_id: int) -> dict:
        tickers = np.asarray(tickers, dtype=np.int64)
        if len(tickers) != task.basket or not np.all(tickers[:-1] < tickers[1:]):
            raise ValueError("A basket must contain K distinct sorted tickers")
        if not np.all(self.streaks[cutoff, tickers] >= task.length):
            raise ValueError("Basket selection used non-historical information or an invalid history")
        labels = labels_from_returns(self.raw[tickers, cutoff:cutoff + task.horizon + 1], 0, task.horizon)
        inputs = np.asarray(self.inputs[tickers, cutoff - task.length + 1:cutoff + 1], dtype=np.float32)
        momentum = np.asarray(self.raw[tickers, cutoff - task.length + 1:cutoff + 1], dtype=np.float64).sum(axis=1)
        ids = np.asarray(self.ids[tickers], dtype=np.int64)
        ids[ids < 0] = unknown_id
        return {"inputs": inputs, "ticker_ids": ids, "market": MARKETS.index(self.market),
                "labels": labels, "observed": np.isfinite(labels), "momentum": momentum,
                "task": task.key, "cutoff": cutoff}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    arguments = parser.parse_args()
    print(prepare(arguments.root, arguments.source), flush=True)


if __name__ == "__main__":
    main()