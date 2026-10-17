import argparse
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from ranker.contract import LENGTHS, MARKETS
from ranker.data import MarketPanel, write_json
from ranker.sampler import coefficients, unrank
from ranking17.recovery import digest


def universe(panel_root, output):
    metadata = json.loads((panel_root / "meta.json").read_text())
    report = {"data_meta_sha256": digest(panel_root / "meta.json"), "dataset_revision": metadata.get("revision"),
              "source_reports_survivors_only": metadata.get("source_reports_survivors_only"),
              "point_in_time_membership_available": False, "delisting_returns_verified": False,
              "known_biases": ["Survivors-only universe can exclude failed and delisted firms.",
                               "Historical input eligibility is not point-in-time exchange membership.",
                               "Label availability may be nonrandom; missing future returns remain unobserved rather than imputed.",
                               "Stock holdout excludes only new child updates; the parent saw the full stock universe.",
                               "Overlapping future-return horizons correlate labels and invalidate independent-basket uncertainty estimates."],
              "label_definition": "Sum raw daily log returns over cutoff+1 through cutoff+horizon; any missing return makes label unobserved.",
              "training_transform": "Training-only scale; clipped normalized history, no future eligibility filter.",
              "verified_source_scope": "Pinned local preparation and sampling code; no external exchange membership or delisting feed audited.",
              "time": time.time()}
    write_json(output / "universe.json", report)
    print(json.dumps({"event": "universe_audit_complete", **report}), flush=True)


def exposure(workspace, panel_root, output):
    candidates = {"1.7.1": workspace / "artifacts/job-286067.hunter-pbs01",
                  "1.9": workspace / "experiment19/artifacts/job-286615.hunter-pbs01",
                  "1.9.1": workspace / "experiment191/artifacts/job-287139.hunter-pbs01"}
    report = {"scope": "Cumulative basket counter ledgers at retained latest checkpoints; not additive across inherited lineages.",
              "1.7": {"status": "unavailable", "reason": "No verified original 1.7 ledger available in this workspace."},
              "experiments": {}, "complete": False}
    for experiment, root in candidates.items():
        pointer = root / "latest.json"
        if not pointer.is_file():
            report["experiments"][experiment] = {"status": "unavailable", "reason": "No verified latest pointer"}
            continue
        checkpoint = (root / json.loads(pointer.read_text())["directory"]).resolve()
        checkpoint.relative_to((root / "checkpoints").resolve())
        ledger_path = checkpoint / "sampler.sqlite"
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if digest(ledger_path) != manifest["sampler.sqlite"]["sha256"]:
            raise ValueError("Exposure audit ledger digest mismatch")
        config = json.loads((checkpoint / "config.json").read_text())
        summary = {"checkpoint": checkpoint.name, "sampler_sha256": digest(ledger_path), "markets": {},
                   "interpretation": "Exact counter-ledger issued exposure; not proof of optimizer consumption if source progress is inconsistent.",
                   "state": json.loads((checkpoint / "state.json").read_text())}
        report["experiments"][experiment] = summary
        with sqlite3.connect(ledger_path.as_uri() + "?mode=ro&immutable=1", uri=True) as ledger:
            for market_index, market in enumerate(MARKETS):
                panel = MarketPanel(panel_root, market)
                stocks, dates = panel.raw.shape
                unique = np.zeros((stocks, dates, 3), dtype=np.uint32)
                differences = np.zeros((stocks, dates + 1), dtype=np.int32)
                baskets, label_uses, observed_uses, total_days = 0, 0, 0, 0
                for key, text in ledger.execute("SELECT stratum, next_counter FROM counters WHERE stratum>=? AND stratum<?",
                                               (market_index << 36, (market_index + 1) << 36)):
                    cutoff, length, horizon, basket = (key >> 20) & 65535, LENGTHS[(key >> 14) & 63], ((key >> 7) & 127) + 1, (key & 127) + 3
                    eligible = panel.eligible(cutoff, length)
                    total, count = math.comb(len(eligible), basket), int(text)
                    if count > total:
                        raise ValueError("Ledger exceeds the original combination universe")
                    multiplier, offset = coefficients(config["model"]["seed"], key, total)
                    for index in range(count):
                        tickers = eligible[list(unrank(len(eligible), basket, (multiplier * index + offset) % total if total > 1 else 0))]
                        baskets += 1
                        label_uses += basket
                        observed = tickers[panel.streaks[cutoff + horizon, tickers] >= horizon]
                        observed_uses += len(observed)
                        total_days += len(observed) * horizon
                        unique[observed, cutoff, (horizon - 1) // 32] |= np.uint32(1 << ((horizon - 1) % 32))
                        differences[observed, cutoff + 1] += 1
                        differences[observed, cutoff + horizon + 1] -= 1
                unique_labels, unique_days = 0, 0
                lookup = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
                for start in range(0, stocks, 128):
                    unique_labels += int(lookup[unique[start:start + 128].view(np.uint8)].sum(dtype=np.uint64))
                    unique_days += int(np.count_nonzero(np.cumsum(differences[start:start + 128], axis=1, dtype=np.int64) > 0))
                summary["markets"][market] = {"baskets": baskets, "scheduled_label_uses": label_uses,
                    "observed_label_uses": observed_uses, "unique_observed_stock_cutoff_horizon": unique_labels,
                    "repeated_observed_label_uses": observed_uses - unique_labels, "observed_label_stock_day_uses": total_days,
                    "unique_observed_label_stock_days": unique_days, "overlapping_label_stock_day_uses": total_days - unique_days}
                del unique, differences
                write_json(output / "exposure.json", report)
                print(json.dumps({"event": "exposure_market_complete", "experiment": experiment, "market": market,
                                  **summary["markets"][market]}), flush=True)
    report["complete"], report["finished_at"] = True, time.time()
    write_json(output / "exposure.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("universe", "exposure"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.action == "universe":
        universe(arguments.panel, arguments.output)
    else:
        exposure(arguments.workspace, arguments.panel, arguments.output)