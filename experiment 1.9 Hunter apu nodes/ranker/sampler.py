from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3

import numpy as np

from .contract import LENGTHS, REVISION, Task, tasks
from .data import MarketPanel


def coefficients(seed: int, stratum: int, size: int) -> tuple[int, int]:
    if size <= 1:
        return 0, 0
    digest = hashlib.sha256(f"{seed}:{stratum}".encode()).digest()
    generator = random.Random(int.from_bytes(digest, "big"))
    multiplier = generator.randrange(1, size)
    while math.gcd(multiplier, size) != 1:
        multiplier = multiplier % (size - 1) + 1
    return multiplier, generator.randrange(size)


def unrank(n: int, k: int, rank: int) -> tuple[int, ...]:
    if not 0 <= k <= n or not 0 <= rank < math.comb(n, k):
        raise ValueError("Combination rank is out of bounds")
    chosen = []
    start = 0
    while k:
        total = math.comb(n - start, k)
        low, high = start, n - k + 1
        while low + 1 < high:
            middle = (low + high) // 2
            before = total - math.comb(n - middle, k)
            if before <= rank:
                low = middle
            else:
                high = middle
        rank -= total - math.comb(n - low, k)
        chosen.append(low)
        start = low + 1
        k -= 1
    return tuple(chosen)


def stratum_id(market: int, cutoff: int, task: Task) -> int:
    if not 0 <= market < 16 or not 0 <= cutoff < 65536:
        raise ValueError("Stratum exceeds the registered market/day index range")
    return (((((market << 16) | cutoff) << 6 | LENGTHS.index(task.length)) << 7
             | (task.horizon - 1)) << 7 | (task.basket - 3))


@dataclass(frozen=True)
class Assignment:
    market: int
    cutoff: int
    task: tuple[int, int, int]
    tickers: tuple[int, ...]

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps((REVISION, self.market, self.cutoff, *self.task, self.tickers),
                                         separators=(",", ":")).encode()).hexdigest()


class Sampler:
    def __init__(self, panels: list[MarketPanel], path: Path, seed: int = 1337, fixed: Task | None = None):
        self.panels = panels
        self.seed = seed
        self.fixed = fixed
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS counters (stratum INTEGER PRIMARY KEY, next_counter TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY CHECK (id=1), cycle INTEGER, position INTEGER, issued INTEGER)")
        self.max_counts = []
        for panel in panels:
            train_boundary = np.searchsorted(panel.dates, np.datetime64("2018-12-31"), side="right") - 1
            maxima = np.zeros((len(LENGTHS), 90), dtype=np.uint16)
            for horizon in range(1, 91):
                endpoint = max(0, train_boundary - horizon + 1)
                if endpoint:
                    maxima[:, horizon - 1] = panel.counts[:, :endpoint].max(axis=1)
            self.max_counts.append(maxima)
        candidates = (fixed,) if fixed else tasks()
        self.catalogue = [task for task in candidates if any(
            maximum[LENGTHS.index(task.length), task.horizon - 1] >= task.basket for maximum in self.max_counts)]
        if not self.catalogue:
            raise ValueError("No feasible historical-input training tasks")
        saved = self.connection.execute("SELECT cycle, position, issued FROM progress WHERE id=1").fetchone()
        self.cycle, self.position, self.issued = saved if saved else (0, 0, 0)
        self.order = self._order()

    def _order(self) -> np.ndarray:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.cycle])).permutation(len(self.catalogue))

    def _sample(self, task: Task, visit: int, basket_number: int) -> Assignment:
        possible = [index for index, maximum in enumerate(self.max_counts)
                    if maximum[LENGTHS.index(task.length), task.horizon - 1] >= task.basket]
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.cycle, visit, basket_number]))
        for _ in range(128):
            market = int(generator.choice(possible))
            panel = self.panels[market]
            days = panel.cutoffs(task, "train")
            if not len(days):
                raise ValueError("Feasibility index disagrees with historical-input counts")
            cutoff = int(generator.choice(days))
            eligible = panel.eligible(cutoff, task.length)
            total = math.comb(len(eligible), task.basket)
            key = stratum_id(market, cutoff, task)
            row = self.connection.execute("SELECT next_counter FROM counters WHERE stratum=?", (key,)).fetchone()
            counter = int(row[0]) if row else 0
            if counter >= total:
                continue
            multiplier, offset = coefficients(self.seed, key, total)
            rank = (multiplier * counter + offset) % total if total > 1 else 0
            tickers = tuple(int(eligible[index]) for index in unrank(len(eligible), task.basket, rank))
            self.connection.execute("INSERT INTO counters (stratum, next_counter) VALUES (?, ?) "
                                    "ON CONFLICT(stratum) DO UPDATE SET next_counter=excluded.next_counter",
                                    (key, str(counter + 1)))
            return Assignment(market, cutoff, task.key, tickers)
        raise RuntimeError("Selected training task repeatedly hit exhausted strata; audit task coverage")

    def issue(self, count: int) -> list[Assignment]:
        if count <= 0 or count % 4:
            raise ValueError("Scheduled samples must consist of complete four-basket task visits")
        result = []
        with self.connection:
            for _ in range(count // 4):
                if self.position == len(self.order):
                    self.cycle += 1
                    self.position = 0
                    self.order = self._order()
                visit = self.position
                task = self.catalogue[int(self.order[visit])]
                for basket_number in range(4):
                    result.append(self._sample(task, visit, basket_number))
                self.position += 1
            self.issued += count
            self.connection.execute("INSERT INTO progress (id, cycle, position, issued) VALUES (1, ?, ?, ?) "
                                    "ON CONFLICT(id) DO UPDATE SET cycle=excluded.cycle, position=excluded.position, issued=excluded.issued",
                                    (self.cycle, self.position, self.issued))
        return result

    def save(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as snapshot:
            self.connection.backup(snapshot)

    def close(self) -> None:
        self.connection.close()

    @classmethod
    def restore(cls, panels: list[MarketPanel], snapshot: Path, destination: Path,
                seed: int = 1337, fixed: Task | None = None) -> Sampler:
        if destination.exists() or Path(str(destination) + "-wal").exists():
            raise ValueError("Resume needs a new sampler ledger path; preserve the old replay log")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, destination)
        return cls(panels, destination, seed, fixed)


def main() -> None:
    import argparse
    from .contract import MARKETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--count", type=int, default=320)
    arguments = parser.parse_args()
    panels = [MarketPanel(arguments.panel, market) for market in MARKETS]
    sampler = Sampler(panels, arguments.ledger)
    print(json.dumps({"feasible_tasks": len(sampler.catalogue), "scheduled": len(sampler.issue(arguments.count))}))
    sampler.close()


if __name__ == "__main__":
    main()