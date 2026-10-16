import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import time

import numpy as np
from safetensors.torch import load_file
import torch
from torch import distributed as dist

from ascgpu.model import ParallelContext
from ranker.contract import MARKETS, Task
from ranker.data import MarketPanel, write_json
from ranker.evaluate import basket_metrics, macro_spearman
from ranker.sampler import Assignment, coefficients, stratum_id, unrank
from ranking17.model import ModelConfig, RankingModel
from ranking17.recovery import digest, verify


def score_sample(model, sample, identity_masked):
    device = model.context.device
    model.eval()
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), precision:
        scores = model(torch.as_tensor(sample["inputs"], device=device)[None],
                       torch.as_tensor(sample["ticker_ids"], device=device)[None],
                       torch.tensor([sample["market"]], device=device),
                       torch.tensor([sample["task"]], device=device),
                       identity_masked=identity_masked)[0][0]
    return scores.float().cpu().numpy()


def novel_assignment(panel, market, cutoff, task, ledger, seed):
    eligible = panel.eligible(cutoff, task.length)
    total = math.comb(len(eligible), task.basket)
    key = stratum_id(market, cutoff, task)
    row = ledger.execute("SELECT next_counter FROM counters WHERE stratum=?", (key,)).fetchone()
    counter = int(row[0]) if row else 0
    if counter >= total:
        raise ValueError("No unused basket remains for this training-period stratum")
    multiplier, offset = coefficients(seed, key, total)
    combination = (multiplier * counter + offset) % total if total > 1 else 0
    tickers = tuple(int(eligible[index]) for index in unrank(len(eligible), task.basket, combination))
    return Assignment(market, cutoff, task.key, tickers)


def paired_summary(first, second):
    cells = defaultdict(list)
    if len(first) != len(second):
        raise ValueError("Paired evaluation row counts differ")
    for before, after in zip(first, second):
        if any(before[key] != after[key] for key in ("index", "task", "market", "cutoff")):
            raise ValueError("Paired evaluation baskets differ")
        if before["spearman"] is not None and after["spearman"] is not None:
            cells[(tuple(before["task"]), before["market"])].append(after["spearman"] - before["spearman"])
    tasks, markets = defaultdict(list), defaultdict(list)
    for (task, market), values in cells.items():
        tasks[task].append(float(np.mean(values)))
        markets[market].append(float(np.mean(values)))
    if not cells:
        raise ValueError("No valid paired correlations")
    market_values = np.asarray([np.mean(values) for values in markets.values()])
    generator = np.random.default_rng(1915411)
    replicates = generator.choice(market_values, (5000, len(market_values)), replace=True).mean(1)
    return {"second_minus_first": float(np.mean([np.mean(values) for values in tasks.values()])),
            "positive_market_task_cells": int(sum(np.mean(values) > 0 for values in cells.values())),
            "market_task_cells": len(cells),
            "by_market": {market: float(np.mean(values)) for market, values in markets.items()},
            "descriptive_market_bootstrap_95pct": np.quantile(replicates, [0.025, 0.975]).tolist(),
            "uncertainty_note": "Market-cluster bootstrap; globally correlated markets and overlapping horizons limit inference."}


def prepare(arguments, panels):
    done = json.loads((arguments.root / "TRAINING_DONE.json").read_text())
    if not done["checkpoint_complete"] or done["step"] != 5411:
        raise ValueError("Diagnostics require the completed step-5411 stop checkpoint")
    checkpoints = {}
    for label, step in (("best", 328), ("latest", 5411)):
        pointer = json.loads((arguments.root / "exports" / f"{label}.json").read_text())
        directory = (arguments.root / "exports" / pointer["directory"]).resolve()
        directory.relative_to((arguments.root / "exports").resolve())
        manifest = verify(directory)
        if set(manifest) != {"config.json", "state.json", "weights-00.safetensors"}:
            raise ValueError("Unexpected inference checkpoint contents")
        state = json.loads((directory / "state.json").read_text())
        if state["step"] != step:
            raise ValueError("Checkpoint pointer changed")
        checkpoints[label] = {"directory": str(directory), "step": step,
                              "manifest_sha256": digest(directory / "manifest.json"),
                              "weights_sha256": manifest["weights-00.safetensors"]["sha256"]}
    config = json.loads((arguments.root / "config.json").read_text())
    for name, key in (("meta.json", "data_meta_sha256"), ("vocabulary.json", "vocabulary_sha256")):
        if digest(arguments.panel / name) != config[key]:
            raise ValueError("Pinned data changed")
    registry = json.loads((arguments.panel / "evaluation/val-primary.json").read_text())
    manifest_path = arguments.panel / "evaluation" / registry["file"]
    if digest(manifest_path) != config["validation_sha256"]:
        raise ValueError("Pinned validation manifest changed")
    allowed = set(registry["monitor_line_indices"])
    validation = [json.loads(line) for index, line in enumerate(manifest_path.read_text().splitlines())
                  if index in allowed]
    if len(validation) != len(allowed) or not validation:
        raise ValueError("Incomplete validation monitoring subset")
    latest = json.loads((arguments.root / "latest.json").read_text())
    ledger_path = (arguments.root / latest["directory"] / "sampler.sqlite").resolve()
    ledger_path.relative_to((arguments.root / "checkpoints").resolve())
    ledger_sha256 = digest(ledger_path)
    training = []
    counts = Counter((tuple(item["task"]), item["market"]) for item in validation)
    with sqlite3.connect(ledger_path.as_uri() + "?mode=ro&immutable=1", uri=True) as ledger:
        for (values, market), count in sorted(counts.items()):
            task, panel = Task(*values), panels[market]
            days = panel.cutoffs(task, "train")
            days = days[panel.dates[days] >= np.datetime64("2015-01-01")]
            if len(days) < count:
                raise ValueError("Insufficient eligible training-era dates for matched control")
            for cutoff in days[np.linspace(0, len(days) - 1, count, dtype=np.int64)]:
                assignment = novel_assignment(panel, market, int(cutoff), task, ledger, config["model"]["seed"])
                training.append(assignment.__dict__)
    if digest(ledger_path) != ledger_sha256:
        raise ValueError("Sampler snapshot changed during evaluation preparation")
    cohorts = {"validation_2019_2022": validation, "training_era_2015_2018_control": training}
    metadata = {"checkpoints": checkpoints, "config": config, "validation_sha256": registry["sha256"],
                "tasks": registry["tasks"], "ledger_path": str(ledger_path), "ledger_sha256": ledger_sha256,
                "true_training_era_holdout": False,
                "training_control_note": "Unused exact baskets in the child sampler, not unseen stocks/dates or independent labels; inherited parent exposure is not excluded.",
                "cohorts": {name: {"baskets": len(items), "earliest_cutoff": min(str(panels[item["market"]].dates[item["cutoff"]]) for item in items),
                                     "latest_cutoff": max(str(panels[item["market"]].dates[item["cutoff"]]) for item in items)}
                            for name, items in cohorts.items()}}
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output / "private-cohorts.json", cohorts)
    write_json(arguments.output / "protocol.json", metadata)
    return metadata, cohorts


def summarize(rows, tasks):
    result = macro_spearman(rows, tasks)
    result["by_year"] = {year: macro_spearman([row for row in rows if row["date"][:4] == year], tasks)["macro_spearman"]
                         for year in sorted({row["date"][:4] for row in rows})}
    return result


def summarize_saved(output, evaluation_source):
    metadata = json.loads((output / "protocol.json").read_text())
    if digest(Path(metadata["ledger_path"])) != metadata["ledger_sha256"]:
        raise ValueError("Sampler changed since diagnostic preparation")
    for checkpoint in metadata["checkpoints"].values():
        directory = Path(checkpoint["directory"])
        manifest = verify(directory)
        if (digest(directory / "manifest.json") != checkpoint["manifest_sha256"]
                or manifest["weights-00.safetensors"]["sha256"] != checkpoint["weights_sha256"]):
            raise ValueError("Diagnostic checkpoint changed")
    tasks = tuple(map(tuple, metadata["tasks"]))
    all_rows, results, hashes = {}, {}, {}
    for label in ("best", "latest"):
        all_rows[label], results[label] = {}, {}
        for cohort, specification in metadata["cohorts"].items():
            path = output / f"rows-{label}-{cohort}.json"
            rows = json.loads(path.read_text())
            if set(rows) != {"ids_on", "ids_hidden", "momentum"}:
                raise ValueError("Incomplete saved diagnostic conditions")
            if any([row["index"] for row in values] != list(range(specification["baskets"]))
                   for values in rows.values()):
                raise ValueError("Saved diagnostic rows are missing or duplicated")
            all_rows[label][cohort] = rows
            hashes[path.name] = digest(path)
            results[label][cohort] = {condition: summarize(values, tasks) for condition, values in rows.items()}
            results[label][cohort]["id_hidden_effect"] = paired_summary(rows["ids_on"], rows["ids_hidden"])
    workers = []
    for path in sorted((output / "logs").rglob("stdout.log")):
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                event = json.loads(line)
                if event.get("event") == "diagnostic_worker_ready":
                    workers.append({key: event[key] for key in ("rank", "host", "pid")})
    if sorted(worker["rank"] for worker in workers) != list(range(32)) or len({worker["host"] for worker in workers}) != 8:
        raise ValueError("Incomplete diagnostic worker roster")
    changes = {}
    for cohort in metadata["cohorts"]:
        if all_rows["best"][cohort]["momentum"] != all_rows["latest"][cohort]["momentum"]:
            raise ValueError("Momentum control changed between checkpoints")
        changes[cohort] = {condition: paired_summary(all_rows["best"][cohort][condition], all_rows["latest"][cohort][condition])
                           for condition in ("ids_on", "ids_hidden")}
    report = {"job": "287139.hunter-pbs01", "evaluation_only": True, "optimizer_created": False,
              "sampler_unchanged": True, "weight_format": "verified BF16 inference exports; BF16 autocast",
              "protocol": {key: value for key, value in metadata.items() if key != "config"},
              "results": results, "latest_minus_best": changes, "workers": sorted(workers, key=lambda worker: worker["rank"]),
              "recovered_from_saved_rows": True, "row_sha256": hashes,
              "source_sha256": digest(evaluation_source), "summarizer_sha256": digest(Path(__file__))}
    write_json(output / "results.json", report)
    return report


def main(arguments):
    if os.environ.get("PBS_JOBID") != "287139.hunter-pbs01":
        raise ValueError("Use the retained job-287139 allocation")
    local_rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, set(range(local_rank * 24, (local_rank + 1) * 24)))
    torch.set_num_threads(4)
    context = ParallelContext.initialize(1)
    if context.dp_size != 32:
        raise ValueError("Expected 32 evaluation workers across eight retained nodes")
    if local_rank == 0:
        active_source = str(arguments.root.parents[1] / "source-287139/train191.py").encode()
        for process in Path("/proc").glob("[0-9]*"):
            try:
                if active_source in (process / "cmdline").read_bytes().split(bytes([0])):
                    raise RuntimeError("Training workers must exit before diagnostic evaluation")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    torch.use_deterministic_algorithms(True)
    panels = [MarketPanel(arguments.panel, market) for market in MARKETS]
    prepared = [None]
    if context.rank == 0:
        try:
            prepared[0] = prepare(arguments, panels)
        except Exception as error:
            prepared[0] = {"error": str(error)}
    dist.broadcast_object_list(prepared, src=0)
    if isinstance(prepared[0], dict):
        raise RuntimeError(prepared[0]["error"])
    metadata, cohorts = prepared[0]
    tasks = tuple(map(tuple, metadata["tasks"]))
    model = RankingModel(ModelConfig(**metadata["config"]["model"]), context)
    model.requires_grad_(False)
    if model.parameter_count() != metadata["config"]["parameters"]:
        raise ValueError("Checkpoint architecture mismatch")
    all_rows, results = {}, {}
    worker = {"rank": context.rank, "host": socket.gethostname(), "pid": os.getpid()}
    print(json.dumps({"event": "diagnostic_worker_ready", **worker}), flush=True)
    started = time.monotonic()
    for label in ("best", "latest"):
        directory = Path(metadata["checkpoints"][label]["directory"])
        model.load_state_dict(load_file(directory / "weights-00.safetensors"))
        results[label], all_rows[label] = {}, {}
        for cohort, items in cohorts.items():
            rows = {condition: [] for condition in ("ids_on", "ids_hidden", "momentum")}
            for index, item in enumerate(items):
                if index % context.dp_size != context.dp_rank:
                    continue
                panel = panels[item["market"]]
                sample = panel.sample(item["cutoff"], Task(*item["task"]), item["tickers"], model.config.tickers)
                common = {"index": index, "task": list(sample["task"]), "market": panel.market,
                          "cutoff": item["cutoff"], "date": str(panel.dates[item["cutoff"]])}
                for condition, masked in (("ids_on", False), ("ids_hidden", True)):
                    scores = score_sample(model, sample, masked)
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite diagnostic scores")
                    rows[condition].append({**common, **basket_metrics(scores, sample["labels"])})
                rows["momentum"].append({**common, **basket_metrics(sample["momentum"], sample["labels"])})
            received = [None] * context.dp_size if context.rank == 0 else None
            dist.gather_object(rows, received, dst=0, group=context.dp_group)
            if context.rank == 0:
                combined = {condition: sorted([row for part in received for row in part[condition]], key=lambda row: row["index"])
                            for condition in rows}
                if any([row["index"] for row in values] != list(range(len(items))) for values in combined.values()):
                    raise ValueError("Distributed diagnostic rows are missing or duplicated")
                all_rows[label][cohort] = combined
                results[label][cohort] = {condition: summarize(values, tasks) for condition, values in combined.items()}
                results[label][cohort]["id_hidden_effect"] = paired_summary(combined["ids_on"], combined["ids_hidden"])
                write_json(arguments.output / f"rows-{label}-{cohort}.json", combined)
                print(json.dumps({"event": "diagnostic_cohort_complete", "checkpoint": label, "cohort": cohort,
                                  "macro_spearman": {condition: values["macro_spearman"] for condition, values in results[label][cohort].items()
                                                     if "macro_spearman" in values}}), flush=True)
    workers = [None] * context.dp_size if context.rank == 0 else None
    dist.gather_object(worker, workers, dst=0, group=context.dp_group)
    if context.rank == 0:
        if digest(Path(metadata["ledger_path"])) != metadata["ledger_sha256"]:
            raise ValueError("Sampler changed during diagnostics")
        changes = {cohort: {condition: paired_summary(all_rows["best"][cohort][condition], all_rows["latest"][cohort][condition])
                            for condition in ("ids_on", "ids_hidden")} for cohort in cohorts}
        report = {"job": "287139.hunter-pbs01", "evaluation_only": True, "optimizer_created": False,
                  "sampler_unchanged": True, "weight_format": "verified BF16 inference exports; BF16 autocast",
                  "protocol": {key: value for key, value in metadata.items() if key != "config"},
                  "results": results, "latest_minus_best": changes, "workers": workers,
                  "seconds": time.monotonic() - started, "source_sha256": digest(Path(__file__))}
        write_json(arguments.output / "results.json", report)
        print(json.dumps({"event": "diagnostics_complete", "results": str(arguments.output / "results.json"),
                          "seconds": report["seconds"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--evaluation-source", type=Path)
    arguments = parser.parse_args()
    if arguments.summarize_only:
        if arguments.evaluation_source is None:
            parser.error("--summarize-only requires --evaluation-source")
        summarize_saved(arguments.output, arguments.evaluation_source)
    else:
        main(arguments)