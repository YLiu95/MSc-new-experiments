import math
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from ranker.contract import LENGTHS
from ranker.data import MarketPanel
from ranker.sampler import Assignment, Sampler, coefficients, stratum_id, unrank
from policy192 import market_probabilities


def combination_rank(size, positions):
    positions = tuple(int(position) for position in positions)
    if not positions or tuple(sorted(set(positions))) != positions or positions[0] < 0 or positions[-1] >= size:
        raise ValueError("Invalid combination positions")
    rank, start, remaining = 0, 0, len(positions)
    for position in positions:
        rank += math.comb(size - start, remaining) - math.comb(size - position, remaining)
        start, remaining = position + 1, remaining - 1
    return rank


def parent_used(original_eligible, tickers, key, counter, seed):
    if not counter:
        return False
    positions = np.searchsorted(original_eligible, tickers)
    if np.any(positions >= len(original_eligible)) or not np.array_equal(original_eligible[positions], tickers):
        raise ValueError("Filtered basket is outside the parent eligibility universe")
    total = math.comb(len(original_eligible), len(tickers))
    if total == 1:
        return True
    multiplier, offset = coefficients(seed, key, total)
    prior_index = ((combination_rank(len(original_eligible), positions) - offset) * pow(multiplier, -1, total)) % total
    return prior_index < counter


class FilteredPanel(MarketPanel):
    def __init__(self, root, market, split_root, cohort="included"):
        super().__init__(root, market)
        if cohort not in ("included", "held_out"):
            raise ValueError("Unknown stock cohort")
        held_out = np.load(Path(split_root) / market / "held_out.npy", allow_pickle=False)
        self.selected = ~held_out if cohort == "included" else held_out
        self.counts = np.load(Path(split_root) / market / f"{cohort}_counts.npy", mmap_mode="r", allow_pickle=False)

    def eligible(self, cutoff, length):
        return np.flatnonzero((self.streaks[cutoff] >= length) & self.selected)


class Sampler192(Sampler):
    def __init__(self, panels, originals, path, arm, weights, seed=1337, fixed=None):
        self.originals, self.arm, self.weights = originals, arm, weights
        super().__init__(panels, path, seed, fixed=fixed)
        self.connection.execute("CREATE TABLE IF NOT EXISTS parent_counters (stratum INTEGER PRIMARY KEY, next_counter TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS exposure (market INTEGER PRIMARY KEY, baskets INTEGER, scheduled_stock_days INTEGER, observed_label_stock_days INTEGER)")
        if any(not np.array_equal(panel.dates, original.dates) for panel, original in zip(panels, originals)):
            raise ValueError("Filtering must preserve the parent's market calendar")
        if self.position > len(self.catalogue):
            raise ValueError("Parent task position exceeds the filtered catalogue")

    @classmethod
    def from_parent(cls, panels, originals, snapshot, destination, arm, weights, seed=1337, fixed=None):
        destination = Path(destination)
        if destination.exists():
            raise ValueError("Sampler destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, destination)
        with sqlite3.connect(destination) as connection:
            connection.execute("ALTER TABLE counters RENAME TO parent_counters")
            connection.execute("CREATE TABLE counters (stratum INTEGER PRIMARY KEY, next_counter TEXT NOT NULL)")
        return cls(panels, originals, destination, arm, weights, seed, fixed)

    @classmethod
    def from_checkpoint(cls, panels, originals, snapshot, destination, arm, weights, seed=1337, fixed=None):
        destination = Path(destination)
        if destination.exists():
            raise ValueError("Sampler destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, destination)
        return cls(panels, originals, destination, arm, weights, seed, fixed)

    def _sample(self, task, visit, basket_number):
        possible = [index for index, maximum in enumerate(self.max_counts)
                    if maximum[LENGTHS.index(task.length), task.horizon - 1] >= task.basket]
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.cycle, visit, basket_number]))
        probabilities = market_probabilities(self.arm, possible, self.weights)
        for _ in range(128):
            market = int(generator.choice(possible) if probabilities is None else generator.choice(possible, p=probabilities))
            panel = self.panels[market]
            cutoff = int(generator.choice(panel.cutoffs(task, "train")))
            eligible = panel.eligible(cutoff, task.length)
            total = math.comb(len(eligible), task.basket)
            key = stratum_id(market, cutoff, task)
            row = self.connection.execute("SELECT next_counter FROM counters WHERE stratum=?", (key,)).fetchone()
            counter = int(row[0]) if row else 0
            previous = self.connection.execute("SELECT next_counter FROM parent_counters WHERE stratum=?", (key,)).fetchone()
            previous = int(previous[0]) if previous else 0
            multiplier, offset = coefficients(self.seed, key, total)
            while counter < total:
                rank = (multiplier * counter + offset) % total if total > 1 else 0
                tickers = tuple(int(eligible[position]) for position in unrank(len(eligible), task.basket, rank))
                counter += 1
                self.connection.execute("INSERT INTO counters VALUES (?, ?) ON CONFLICT(stratum) DO UPDATE SET next_counter=excluded.next_counter",
                                        (key, str(counter)))
                if not parent_used(self.originals[market].eligible(cutoff, task.length), tickers, key, previous, self.seed):
                    return Assignment(market, cutoff, task.key, tickers)
        raise RuntimeError("Filtered sampler exhausted feasible strata")

    def issue(self, count):
        result = super().issue(count)
        totals = {}
        for assignment in result:
            horizon = assignment.task[1]
            observed = int(np.count_nonzero(self.panels[assignment.market].streaks[
                assignment.cutoff + horizon, list(assignment.tickers)] >= horizon))
            values = totals.setdefault(assignment.market, [0, 0, 0])
            values[0] += 1
            values[1] += len(assignment.tickers) * horizon
            values[2] += observed * horizon
        with self.connection:
            for market, values in totals.items():
                self.connection.execute("INSERT INTO exposure VALUES (?, ?, ?, ?) ON CONFLICT(market) DO UPDATE SET "
                                        "baskets=baskets+excluded.baskets, scheduled_stock_days=scheduled_stock_days+excluded.scheduled_stock_days, "
                                        "observed_label_stock_days=observed_label_stock_days+excluded.observed_label_stock_days",
                                        (market, *values))
        return result

    def exposure(self):
        return {self.panels[market].market: {"baskets": baskets, "scheduled_stock_days": scheduled,
                                           "observed_label_stock_days": observed}
                for market, baskets, scheduled, observed in self.connection.execute("SELECT * FROM exposure")}