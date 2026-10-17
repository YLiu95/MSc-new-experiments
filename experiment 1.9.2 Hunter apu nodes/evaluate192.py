import json
from pathlib import Path

import numpy as np
from torch import distributed as dist

from diagnose191 import score_sample
from ranker.contract import Task
from ranker.data import write_json
from ranker.evaluate import basket_metrics, macro_spearman
from ranking17.recovery import digest, verify


def evaluate_cohort(model, panels, items, tasks, identity_masked=True):
    context = model.context
    rows = []
    for index, item in enumerate(items):
        if index % context.dp_size != context.dp_rank:
            continue
        panel = panels[item["market"]]
        sample = panel.sample(item["cutoff"], Task(*item["task"]), item["tickers"], model.config.tickers)
        scores = score_sample(model, sample, identity_masked)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite evaluation scores")
        rows.append({"index": index, "task": list(sample["task"]), "market": panel.market,
                     "cutoff": item["cutoff"], "date": str(panel.dates[item["cutoff"]]),
                     **basket_metrics(scores, sample["labels"])})
    received = [None] * context.dp_size if context.rank == 0 else None
    dist.gather_object(rows, received, dst=0, group=context.dp_group)
    status, combined = [None], None
    if context.rank == 0:
        try:
            combined = sorted([row for part in received for row in part], key=lambda row: row["index"])
            if [row["index"] for row in combined] != list(range(len(items))):
                raise ValueError("Evaluation rows are missing or duplicated")
            summary = macro_spearman(combined, tuple(map(tuple, tasks)))
            summary["by_market"] = {market: float(np.mean([
                np.mean([row["spearman"] for row in combined if row["market"] == market and tuple(row["task"]) == task
                         and row["spearman"] is not None])
                for task in sorted({tuple(row["task"]) for row in combined if row["market"] == market and row["spearman"] is not None})]))
                for market in sorted({row["market"] for row in combined})}
            status[0] = summary
        except Exception as error:
            status[0] = {"error": str(error)}
    dist.broadcast_object_list(status, src=0)
    if "error" in status[0]:
        raise RuntimeError(status[0]["error"])
    model.train()
    return status[0], combined


def require_final_selection(root, final_evaluation, protocol):
    if not final_evaluation:
        raise ValueError("Final test is sealed; explicit --final-evaluation is required")
    if not protocol.get("final_test_unseen_user_confirmed"):
        raise ValueError("Prior test exposure must be resolved before final evaluation")
    selection = {}
    for arm in ("A", "B"):
        arm_root = root / f"arm-{arm}"
        done = json.loads((arm_root / "TRAINING_DONE.json").read_text())
        if not done.get("checkpoint_complete"):
            raise ValueError("Both arms must finish and freeze selection first")
        pointer = json.loads((arm_root / "best.json").read_text())
        directory = (arm_root / pointer["directory"]).resolve()
        directory.relative_to((arm_root / "checkpoints").resolve())
        verify(directory, workers=4)
        selection[arm] = {"directory": str(directory), "manifest_sha256": digest(directory / "manifest.json"),
                          "step": pointer["step"]}
    if (root / "final-test" / "OPENED.json").exists():
        raise ValueError("Final test was already opened; reuse saved rows, do not rerun")
    return selection


def time_block_difference(first, second, block_sessions=90, seed=192):
    if len(first) != len(second):
        raise ValueError("Paired evaluation sizes differ")
    cells = {}
    for before, after in zip(first, second):
        if any(before[key] != after[key] for key in ("index", "market", "task", "cutoff")):
            raise ValueError("Paired evaluation coordinates differ")
        if before["spearman"] is not None and after["spearman"] is not None:
            key = (before["market"], tuple(before["task"]), before["cutoff"] // block_sessions)
            cells.setdefault(key, []).append(after["spearman"] - before["spearman"])
    result = {}
    generator = np.random.default_rng(seed)
    for market in sorted({key[0] for key in cells}):
        blocks = sorted({key[2] for key in cells if key[0] == market})
        tasks = sorted({key[1] for key in cells if key[0] == market})
        matrix = np.asarray([[np.mean(cells[(market, task, block)]) if (market, task, block) in cells else np.nan
                              for block in blocks] for task in tasks])
        replicates = []
        for _ in range(2000):
            sampled = matrix[:, generator.integers(0, len(blocks), len(blocks))]
            if np.all(np.any(np.isfinite(sampled), axis=1)):
                replicates.append(float(np.nanmean(sampled, axis=1).mean()))
        result[market] = {"second_minus_first": float(np.mean([np.mean([value for key, values in cells.items()
                          if key[:2] == (market, task) for value in values]) for task in tasks])),
                          "time_blocks": len(blocks), "block_sessions": block_sessions,
                          "time_block_bootstrap_95pct": np.quantile(replicates, [0.025, 0.975]).tolist() if len(replicates) >= 100 else None,
                          "note": "Paired nonoverlapping session-block resampling within market; exploratory, not a cross-market independence claim."}
    return result