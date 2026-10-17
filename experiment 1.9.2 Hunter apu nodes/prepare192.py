import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from ranker.contract import LENGTHS, MARKETS, Task
from ranker.data import MarketPanel, write_json
from ranker.sampler import Assignment, coefficients, stratum_id, unrank
from ranking17.recovery import digest
from policy192 import is_held_out
from sampler192 import FilteredPanel, parent_used
from train191 import verify_parent


def matched_cohorts(originals, included, held_out, validation, ledger, seed):
    cohorts = {"included": [], "held_out": []}
    cells = []
    for (values, market), requested in sorted(Counter((tuple(item["task"]), item["market"]) for item in validation).items()):
        task = Task(*values)
        days = np.intersect1d(included[market].cutoffs(task, "train"), held_out[market].cutoffs(task, "train"))
        days = days[originals[market].dates[days] >= np.datetime64("2015-01-01")]
        count = min(requested, len(days))
        cells.append({"market": MARKETS[market], "task": list(values), "requested": requested, "actual": count})
        for cutoff in days[np.linspace(0, len(days) - 1, count, dtype=np.int64)] if count else []:
            cutoff = int(cutoff)
            key = stratum_id(market, cutoff, task)
            row = ledger.execute("SELECT next_counter FROM counters WHERE stratum=?", (key,)).fetchone()
            consumed = int(row[0]) if row else 0
            for name, panels in (("included", included), ("held_out", held_out)):
                eligible = panels[market].eligible(cutoff, task.length)
                total = math.comb(len(eligible), task.basket)
                multiplier, offset = coefficients(seed + 192, key, total)
                for counter in range(min(total, consumed + 1)):
                    rank = (multiplier * counter + offset) % total if total > 1 else 0
                    tickers = tuple(int(eligible[position]) for position in unrank(len(eligible), task.basket, rank))
                    if not parent_used(originals[market].eligible(cutoff, task.length), tickers, key, consumed, seed):
                        cohorts[name].append(Assignment(market, cutoff, task.key, tickers).__dict__)
                        break
                else:
                    raise ValueError("Cannot construct an unused matched cohort basket")
    if not cohorts["included"] or len(cohorts["included"]) != len(cohorts["held_out"]):
        raise ValueError("Matched stock holdout cohorts are incomplete")
    return cohorts, cells


def prepare(arguments):
    config, state = verify_parent(arguments.parent)
    root = arguments.output
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, key in (("meta.json", "data_meta_sha256"), ("vocabulary.json", "vocabulary_sha256")):
        if digest(arguments.panel / name) != config[key]:
            raise ValueError("Parent data fingerprint changed")
    registry = json.loads((arguments.panel / "evaluation/val-primary.json").read_text())
    manifest = arguments.panel / "evaluation" / registry["file"]
    if digest(manifest) != config["validation_sha256"]:
        raise ValueError("Parent validation baskets changed")
    allowed = set(registry["monitor_line_indices"])
    validation = [json.loads(line) for index, line in enumerate(manifest.read_text().splitlines()) if index in allowed]
    if len(validation) != 1616:
        raise ValueError("Expected the fixed 1616-basket validation subset")
    originals, private, weights, market_report = [], {}, [], {}
    for market in MARKETS:
        panel = MarketPanel(arguments.panel, market)
        originals.append(panel)
        symbols = json.loads((arguments.panel / market / "symbols.json").read_text())
        mask = np.asarray([is_held_out(market, ticker) for ticker in symbols], dtype=np.bool_)
        directory = root / market
        directory.mkdir(mode=0o700)
        np.save(directory / "held_out.npy", mask, allow_pickle=False)
        endpoint = int(np.searchsorted(panel.dates, np.datetime64("2018-12-31"), side="right"))
        for name, selected in (("included", ~mask), ("held_out", mask)):
            counts = np.zeros_like(panel.counts)
            for start in range(0, endpoint, 256):
                block = panel.streaks[start:min(start + 256, endpoint), selected]
                for index, length in enumerate(LENGTHS):
                    counts[index, start:min(start + 256, endpoint)] = np.count_nonzero(block >= length, axis=1)
            np.save(directory / f"{name}_counts.npy", counts, allow_pickle=False)
            if name == "included":
                weights.append(int(counts[0, :max(0, endpoint - 1)].sum(dtype=np.int64)))
        private[market] = [ticker for ticker, selected in zip(symbols, mask) if selected]
        market_report[market] = {"universe_stocks": len(symbols), "held_out_stocks": int(mask.sum()),
                                 "eligible_training_stock_days": weights[-1]}
        print(json.dumps({"event": "stock_split_prepared", "market": market, **market_report[market]}), flush=True)
    included = [FilteredPanel(arguments.panel, market, root) for market in MARKETS]
    held_out = [FilteredPanel(arguments.panel, market, root, "held_out") for market in MARKETS]
    snapshot = arguments.parent / "sampler.sqlite"
    original_hash = digest(snapshot)
    with sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as ledger:
        cohorts, cells = matched_cohorts(originals, included, held_out, validation, ledger, config["model"]["seed"])
    if digest(snapshot) != original_hash:
        raise ValueError("Parent sampler changed")
    cohorts["validation"] = validation
    write_json(root / "private-cohorts.json", cohorts)
    write_json(root / "private-held-out-stocks.json", private)
    checksums = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*")) if path.is_file()}
    protocol = {"experiment": "1.9.2", "parent_manifest_sha256": digest(arguments.parent / "manifest.json"),
                "parent_scheduled": state["scheduled"], "hash_rule": "big_endian_integer(sha256(UTF8(market+ticker))) * 10 < 2**256",
                "weight_definition": "Included stocks with >=64 historical sessions summed over eligible training cutoffs for horizon 1",
                "market_weights": weights, "markets": market_report, "tasks": registry["tasks"],
                "validation_baskets": len(validation), "matched_baskets_per_cohort": len(cohorts["included"]),
                "matched_cells": cells, "user_approved_feasible_cells_only": True,
                "final_test_unseen_user_confirmed": True, "final_test_opened": False,
                "cohort_limitation": "Only withheld from 1.9.2 updates; parent exposure is not excluded. Included baskets are not a label holdout.",
                "private_file_sha256": checksums, "source_sha256": digest(Path(__file__))}
    write_json(root / "protocol.json", protocol)
    (root / "COMPLETE").write_text(digest(root / "protocol.json") + "\n")
    print(json.dumps({"event": "preparation_complete", "validation": len(validation), "matched": len(cohorts["included"])}), flush=True)


def load_protocol(root):
    if (root / "COMPLETE").read_text().strip() != digest(root / "protocol.json"):
        raise ValueError("Preparation is incomplete or changed")
    protocol = json.loads((root / "protocol.json").read_text())
    for name, expected in protocol["private_file_sha256"].items():
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        if digest(path) != expected:
            raise ValueError("Private cohort or stock split changed")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())