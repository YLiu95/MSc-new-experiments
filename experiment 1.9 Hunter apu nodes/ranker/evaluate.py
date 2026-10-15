from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np
import torch

from .contract import HELD_OUT, MARKETS, PRIMARY, Task
from .data import MarketPanel, write_json
from .sampler import Assignment, coefficients, stratum_id, unrank


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    _, first, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat(first + (counts - 1) / 2 + 1, counts)
    return ranks


def basket_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    observed = np.isfinite(labels)
    actual = labels[observed]
    prediction = scores[observed]
    spearman = None
    constant_scores = len(prediction) > 0 and np.all(prediction == prediction[0])
    constant_returns = len(actual) > 0 and np.all(actual == actual[0])
    if len(actual) >= 3 and not constant_returns:
        ranks = average_ranks(actual)
        spearman = 0.0 if constant_scores else float(np.corrcoef(average_ranks(prediction), ranks)[0, 1])
    upper = np.triu_indices(len(actual), k=1)
    differences = actual[upper[0]] - actual[upper[1]]
    informative = differences != 0
    score_differences = prediction[upper[0]] - prediction[upper[1]]
    correct = float(((np.sign(differences) == np.sign(score_differences)) & informative).sum()
                    + 0.5 * ((score_differences == 0) & informative).sum())
    selected = max(1, int(math.floor(0.2 * len(scores))))
    ordering = np.argsort(scores, kind="stable")
    top = labels[ordering[-selected:]]
    bottom = labels[ordering[:selected]]
    return {"spearman": spearman, "observed": int(observed.sum()), "unobserved": int((~observed).sum()),
            "constant_returns": bool(constant_returns), "constant_scores": bool(constant_scores),
            "pairs": int(informative.sum()), "correct_pairs": correct,
            "top_sum": float(np.nansum(top, dtype=np.float64)), "top_observed": int(np.isfinite(top).sum()),
            "bottom_sum": float(np.nansum(bottom, dtype=np.float64)), "bottom_observed": int(np.isfinite(bottom).sum()),
            "score_variance": float(np.var(scores, dtype=np.float64))}


def macro_spearman(rows: list[dict], required: tuple[tuple[int, int, int], ...]) -> dict:
    within_dates = defaultdict(list)
    cells = defaultdict(list)
    missing = []
    for row in rows:
        if row["spearman"] is not None:
            within_dates[(tuple(row["task"]), row["market"], row["cutoff"])].append(row["spearman"])
    for (task, market, _), values in within_dates.items():
        cells[(task, market)].append(float(np.mean(values, dtype=np.float64)))
    per_task = {}
    for task in required:
        markets = []
        for market in MARKETS:
            values = cells.get((task, market), [])
            if values:
                markets.append(float(np.mean(values, dtype=np.float64)))
            else:
                missing.append({"task": task, "market": market})
        if not markets:
            raise ValueError(f"Primary evaluation task has no observable Spearman: {task}")
        per_task[str(task)] = float(np.mean(markets, dtype=np.float64))
    return {"macro_spearman": float(np.mean(list(per_task.values()), dtype=np.float64)),
            "tasks": per_task, "missing_market_task_cells": missing,
            "degenerate_correlations": sum(row["spearman"] is None for row in rows),
            "baskets": len(rows), "observed_labels": sum(row["observed"] for row in rows),
            "missing_labels": sum(row["unobserved"] for row in rows),
            "informative_pairs": sum(row["pairs"] for row in rows),
            "pair_accuracy": (sum(row["correct_pairs"] for row in rows) /
                              max(1, sum(row["pairs"] for row in rows))),
            "top_observed": sum(row["top_observed"] for row in rows),
            "bottom_observed": sum(row["bottom_observed"] for row in rows),
            "top_mean": (sum(row["top_sum"] for row in rows) /
                         max(1, sum(row["top_observed"] for row in rows))),
            "bottom_mean": (sum(row["bottom_sum"] for row in rows) /
                            max(1, sum(row["bottom_observed"] for row in rows)))}


def build_manifest(directory: Path, panels: list[MarketPanel], split: str, seed: int,
                   cells: tuple[tuple[int, int, int], ...] = PRIMARY,
                   max_cutoffs: int = 128, baskets: int = 4) -> dict:
    if split == "test" and seed != 9331:
        raise ValueError("Test manifests must use the registered post-selection seed")
    directory.mkdir(parents=True, exist_ok=True)
    label = "primary" if cells == PRIMARY else "heldout" if set(cells) == HELD_OUT else "fixture"
    path = directory / f"{split}-{label}.jsonl"
    registry = directory / f"{split}-{label}.json"
    if path.exists():
        saved = json.loads(registry.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (saved["sha256"] != digest or saved["seed"] != seed or saved["tasks"] != [list(cell) for cell in cells]
                or saved["max_cutoffs"] != max_cutoffs or saved["baskets_per_cutoff"] != baskets):
            raise ValueError("Frozen evaluation manifest differs from the registered suite")
        return saved
    temporary = path.with_suffix(".tmp")
    count = 0
    monitoring = []
    counts = {}
    with temporary.open("w") as stream:
        for task_values in cells:
            task = Task(*task_values)
            for market, panel in enumerate(panels):
                eligible_days = panel.cutoffs(task, split)
                selection = np.linspace(0, len(eligible_days) - 1, min(max_cutoffs, len(eligible_days)), dtype=np.int64)
                cutoffs = eligible_days[selection] if len(eligible_days) else []
                cell_count = 0
                for cutoff_position, day in enumerate(cutoffs):
                    cutoff = int(day)
                    eligible_tickers = panel.eligible(cutoff, task.length)
                    combinations = math.comb(len(eligible_tickers), task.basket)
                    multiplier, offset = coefficients(seed, stratum_id(market, cutoff, task), combinations)
                    for index in range(min(baskets, combinations)):
                        rank = (multiplier * index + offset) % combinations if combinations > 1 else 0
                        tickers = tuple(int(eligible_tickers[position]) for position in
                                        unrank(len(eligible_tickers), task.basket, rank))
                        assignment = Assignment(market, cutoff, task.key, tickers)
                        stream.write(json.dumps(assignment.__dict__, separators=(",", ":")) + "\n")
                        count += 1
                        cell_count += 1
                        if index == 0 and cutoff_position in set(np.linspace(0, len(cutoffs) - 1,
                                                                           min(16, len(cutoffs)), dtype=np.int64)):
                            monitoring.append(count - 1)
                counts[f"{task.key}/{panel.market}"] = cell_count
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"file": path.name, "sha256": digest, "seed": seed, "split": split, "tasks": [list(cell) for cell in cells],
              "max_cutoffs": max_cutoffs, "baskets_per_cutoff": baskets,
              "baskets": count, "market_task_counts": counts,
              "monitor_line_indices": monitoring if split == "val" and label == "primary" else []}
    write_json(registry, result)
    if label == "primary" and not all(any(counts[f"{task}/{panel.market}"] for panel in panels) for task in cells):
        raise ValueError("A registered primary task has no historically eligible market")
    return result


def evaluate_model(model: torch.nn.Module, panels: list[MarketPanel], panel_root: Path, directory: Path,
                   registry: dict, device: torch.device, rank: int = 0, world: int = 1,
                   monitoring: bool = False) -> dict | None:
    from torch import distributed as dist

    model.eval()
    lines = (directory / registry["file"]).read_text().splitlines()
    allowed = set(registry["monitor_line_indices"]) if monitoring else None
    unknown = json.loads((panel_root / "meta.json").read_text())["n_tickers"]
    outputs = {name: [] for name in ("model", "momentum", "reversal", "constant")}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast:
        indexed = ((index, json.loads(line)) for index, line in enumerate(lines))
        grouped = groupby(indexed, key=lambda pair: (pair[1]["market"], pair[1]["cutoff"], pair[1]["task"]))
        for group_index, (_, pairs) in enumerate(grouped):
            if group_index % world != rank:
                continue
            selected = [(index, item) for index, item in pairs if allowed is None or index in allowed]
            if not selected:
                continue
            samples = [panels[item["market"]].sample(item["cutoff"], Task(*item["task"]),
                                                       item["tickers"], unknown) for _, item in selected]
            inputs = torch.as_tensor(np.stack([sample["inputs"] for sample in samples]),
                                     dtype=torch.float32, device=device)
            identities = torch.as_tensor(np.stack([sample["ticker_ids"] for sample in samples]),
                                         dtype=torch.long, device=device)
            markets = torch.as_tensor([sample["market"] for sample in samples], dtype=torch.long, device=device)
            tasks_tensor = torch.as_tensor([sample["task"] for sample in samples], dtype=torch.long, device=device)
            predicted = model(inputs, identities, markets, tasks_tensor)[0].float().cpu().numpy()
            for (_, item), sample, scores in zip(selected, samples, predicted):
                for name, predictions in (("model", scores), ("momentum", sample["momentum"]),
                                          ("reversal", -sample["momentum"]),
                                          ("constant", np.zeros(len(scores)))):
                    outputs[name].append({**basket_metrics(predictions, sample["labels"]),
                                          "task": sample["task"], "market": panels[item["market"]].market,
                                          "cutoff": str(panels[item["market"]].dates[sample["cutoff"]])})
    if world > 1:
        received = [None] * world if rank == 0 else None
        dist.gather_object(outputs, received, dst=0)
        if rank != 0:
            model.train()
            return None
        outputs = {name: [row for chunk in received for row in chunk[name]] for name in outputs}
    result = {name: macro_spearman(rows, tuple(map(tuple, registry["tasks"]))) for name, rows in outputs.items()}
    for summary in result.values():
        if summary["top_observed"] and summary["bottom_observed"]:
            summary["top_minus_bottom"] = summary["top_mean"] - summary["bottom_mean"]
    result["manifest_sha256"] = registry["sha256"]
    result["monitoring_only"] = monitoring
    model.train()
    return result


def baseline_report(panels: list[MarketPanel], panel_root: Path, directory: Path, registry: dict) -> dict:
    unknown = json.loads((panel_root / "meta.json").read_text())["n_tickers"]
    groups = defaultdict(lambda: [0, 0])
    observations = {name: [] for name in ("momentum", "reversal", "constant")}
    with (directory / registry["file"]).open() as stream:
        for line in stream:
            item = json.loads(line)
            task = Task(*item["task"])
            panel = panels[item["market"]]
            sample = panel.sample(item["cutoff"], task, item["tickers"], unknown)
            for ticker_id, observed in zip(sample["ticker_ids"], sample["observed"]):
                cohort = "unseen_identity" if ticker_id == unknown else "training_seen_identity"
                counts = groups[(panel.market, task.horizon, cohort)]
                counts[0] += 1
                counts[1] += int(observed)
            for name, scores in (("momentum", sample["momentum"]), ("reversal", -sample["momentum"]),
                                 ("constant", np.zeros(task.basket))):
                observations[name].append({**basket_metrics(scores, sample["labels"]),
                                           "task": task.key, "market": panel.market,
                                           "cutoff": str(panel.dates[sample["cutoff"]])})
    summary = {name: macro_spearman(rows, tuple(map(tuple, registry["tasks"]))) for name, rows in observations.items()}
    summary["missing_by_market_horizon_cohort"] = [
        {"market": market, "horizon": horizon, "cohort": cohort,
         "selected_labels": selected, "observed_labels": observed, "missing_labels": selected - observed}
        for (market, horizon, cohort), (selected, observed) in sorted(groups.items())]
    summary["manifest_sha256"] = registry["sha256"]
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "baselines", "test"))
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    panel_root = arguments.root / "panel"
    panels = [MarketPanel(panel_root, market) for market in MARKETS]
    directory = panel_root / "evaluation"
    if arguments.action == "register":
        primary = build_manifest(directory, panels, "val", 7331)
        heldout = build_manifest(directory, panels, "val", 7331, tuple(sorted(HELD_OUT)))
        write_json(arguments.root / "reports" / "evaluation_registry.json", {
            "primary": {key: value for key, value in primary.items() if key != "monitor_line_indices"},
            "heldout": {key: value for key, value in heldout.items() if key != "monitor_line_indices"}})
        print(json.dumps({"primary_baskets": primary["baskets"], "monitor_baskets": len(primary["monitor_line_indices"]),
                          "heldout_baskets": heldout["baskets"], "primary_sha256": primary["sha256"]}), flush=True)
    elif arguments.action == "baselines":
        registry = json.loads((directory / "val-primary.json").read_text())
        summary = baseline_report(panels, panel_root, directory, registry)
        write_json(arguments.root / "reports" / "baselines.json", summary)
        print(json.dumps({"momentum_macro": summary["momentum"]["macro_spearman"],
                          "reversal_macro": summary["reversal"]["macro_spearman"],
                          "constant_macro": summary["constant"]["macro_spearman"]}), flush=True)
    else:
        if not (arguments.root / "C" / "best.json").is_file():
            raise ValueError("Freeze checkpoint selection before building or inspecting the test suite")
        primary = build_manifest(directory, panels, "test", 9331)
        heldout = build_manifest(directory, panels, "test", 9331, tuple(sorted(HELD_OUT)))
        print(json.dumps({"test_primary_baskets": primary["baskets"],
                          "test_heldout_baskets": heldout["baskets"]}), flush=True)


if __name__ == "__main__":
    main()