import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from safetensors.torch import load_file
import torch
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

from ascgpu.model import ParallelContext
from ranker.contract import MARKETS
from ranker.data import MarketPanel, write_json
from ranker.evaluate import build_manifest
from ranking17.model import ModelConfig
from ranking17.recovery import digest, verify
from evaluate192 import evaluate_cohort, require_final_selection, time_block_difference
from model192 import IdentityHiddenModel
from policy192 import JOB, WORLD
from prepare192 import load_protocol
from train192 import rank_zero
from train191 import verify_parent


TEST_REPORT_SPLIT = {"name": "test-report", "validation_end": "2022-12-31", "embargo_sessions": 90,
                     "report_end": "2024-12-31", "final_holdout_end": "2026-08-25",
                     "final_holdout_sealed": True, "amendment_date": "2026-10-17"}


class TestReportPanel:
    def __init__(self, panel):
        self.panel = panel

    def __getattr__(self, name):
        return getattr(self.panel, name)

    def cutoffs(self, task, split):
        if split != "test":
            raise ValueError("Test-report wrapper is for test access only")
        cutoffs = self.panel.cutoffs(task, split)
        boundary = np.searchsorted(self.dates, np.datetime64(TEST_REPORT_SPLIT["report_end"]), side="right")
        return cutoffs[cutoffs + task.horizon < boundary]


def validate_test_report(items, panels):
    if not items:
        raise ValueError("Test-report manifest is empty")
    for item in items:
        dates = panels[item["market"]].dates
        lower = np.searchsorted(dates, np.datetime64("2022-12-31"), side="right") - 1 + 90
        upper = np.searchsorted(dates, np.datetime64("2024-12-31"), side="right")
        if not lower <= item["cutoff"] or item["cutoff"] + item["task"][1] >= upper:
            raise ValueError("Test-report manifest enters embargo or sealed final holdout")


def selection_identity(selection):
    return {label: {key: checkpoint[key] for key in ("step", "manifest_sha256")}
            for label, checkpoint in selection.items()}


def open_or_resume_test(output, selection, parent, panels):
    identity = {"selection": selection, "parent_manifest_sha256": digest(parent / "manifest.json"),
                "split": TEST_REPORT_SPLIT}
    output.mkdir(parents=True, exist_ok=True)
    opened = output / "OPENED.json"
    frozen = output / "SELECTION.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text())
        if (selection_identity(saved["selection"]) != selection_identity(selection)
                or saved["parent_manifest_sha256"] != identity["parent_manifest_sha256"] or saved["split"] != TEST_REPORT_SPLIT):
            raise ValueError("Frozen final-test selection or protocol changed")
        identity = saved
    else:
        if opened.exists():
            raise ValueError("Opened final test is missing its frozen selection")
        write_json(frozen, identity)
    if opened.exists():
        record = json.loads(opened.read_text())
        if any(record[key] != identity[key] for key in identity):
            raise ValueError("Opened final-test identity changed")
    else:
        manifest_directory = output / "private-manifest"
        if manifest_directory.exists():
            registry = build_manifest(manifest_directory, [TestReportPanel(panel) for panel in panels], "test", 9331)
        else:
            staging = Path(tempfile.mkdtemp(prefix="manifest-staging-", dir=output))
            registry = build_manifest(staging, [TestReportPanel(panel) for panel in panels], "test", 9331)
            validate_test_report([json.loads(line) for line in (staging / registry["file"]).read_text().splitlines()], panels)
            staging.rename(manifest_directory)
        record = {**identity, "registry": registry, "time": time.time(), "user_confirmed_no_previous_test_evaluation": True}
        write_json(opened, record)
    registry = record["registry"]
    path = output / "private-manifest" / registry["file"]
    if digest(path) != registry["sha256"]:
        raise ValueError("Frozen test-report manifest changed")
    items = [json.loads(line) for line in path.read_text().splitlines()]
    validate_test_report(items, panels)
    return record, items


def saved_condition(output, label, mode, checkpoint_hash, registry, items, evaluate, synchronize):
    path = output / f"private-rows-{label}-{mode}.json"
    provenance = {"checkpoint_manifest_sha256": checkpoint_hash, "test_manifest_sha256": registry["sha256"], "mode": mode}

    def validate(record):
        if any(record.get(key) != value for key, value in provenance.items()):
            raise ValueError("Saved final-test provenance differs")
        rows = record["rows"]
        if [row["index"] for row in rows] != list(range(len(items))):
            raise ValueError("Saved final-test rows are incomplete")
        for row, item in zip(rows, items):
            if row["market"] != MARKETS[item["market"]] or row["cutoff"] != item["cutoff"] or row["task"] != item["task"]:
                raise ValueError("Saved final-test row coordinates differ")
        return record

    cached = synchronize(lambda: validate(json.loads(path.read_text())) if path.exists() else None)
    if cached is not None:
        return cached
    summary, rows = evaluate()

    def save():
        record = validate({**provenance, "summary": summary, "rows": rows})
        write_json(path, record)
        return record

    return synchronize(save)


def main(arguments):
    if not arguments.final_evaluation:
        raise ValueError("Final test is sealed; pass --final-evaluation after both arms finish")
    if os.environ.get("PBS_JOBID") != JOB:
        raise ValueError("Use the retained allocation")
    local_rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, set(range(local_rank * 24, (local_rank + 1) * 24)))
    torch.set_num_threads(4)
    context = ParallelContext.initialize(1)
    if context.dp_size != WORLD:
        raise ValueError(f"Final evaluation requires all {WORLD} allocated GPU workers")
    protocol = rank_zero(context, lambda: load_protocol(arguments.prepared))
    selection = rank_zero(context, lambda: require_final_selection(arguments.root, arguments.final_evaluation, protocol,
                                                                 arguments.arm_a_root, resume=True))
    output = arguments.root / "final-test"
    enough_time = rank_zero(context, lambda: time.time() + 3600 < arguments.end_at - 120)
    if not enough_time:
        rank_zero(context, lambda: write_json(arguments.root / "EXPERIMENT_DONE.json", {
            "training_complete": True, "final_test_complete": False, "final_test_opened": (output / "OPENED.json").exists(),
            "reason": "insufficient_time_for_full_final_test", "selection": selection, "time": time.time()}))
        dist.barrier()
        dist.destroy_process_group()
        return
    config, _ = rank_zero(context, lambda: verify_parent(arguments.parent))
    panels = [MarketPanel(arguments.panel, market) for market in MARKETS]
    opened = rank_zero(context, lambda: open_or_resume_test(output, selection, arguments.parent, panels)[0])
    selection, registry = opened["selection"], opened["registry"]
    items = [json.loads(line) for line in (output / "private-manifest" / registry["file"]).read_text().splitlines()]
    validate_test_report(items, panels)
    model = IdentityHiddenModel(ModelConfig(**config["model"]), context)
    model.requires_grad_(False)
    model.eval()
    summaries, all_rows = {"hidden": {}, "ids_on": {}}, {"hidden": {}, "ids_on": {}}
    for label, directory in [("parent-328", arguments.parent)] + [(arm, Path(selection[arm]["directory"])) for arm in ("A", "B")]:
        checkpoint_hash = rank_zero(context, lambda: digest(directory / "manifest.json"))
        for mode in ("hidden", "ids_on"):
            if rank_zero(context, lambda: time.time() + 600 >= arguments.end_at - 120):
                rank_zero(context, lambda: write_json(arguments.root / "EXPERIMENT_DONE.json", {
                    "training_complete": True, "final_test_complete": False, "final_test_opened": True,
                    "reason": "evaluation_checkpoint_reserve", "time": time.time()}))
                dist.barrier()
                dist.destroy_process_group()
                return

            def evaluate():
                rank_zero(context, lambda: verify(directory, ("weights-00.safetensors", "config.json", "state.json")))
                model.load_state_dict(load_file(directory / "weights-00.safetensors"))
                model.eval()
                return evaluate_cohort(model, panels, items, registry["tasks"], mode == "hidden")

            record = saved_condition(output, label, mode, checkpoint_hash, registry, items, evaluate,
                                     lambda operation: rank_zero(context, operation))
            summaries[mode][label], all_rows[mode][label] = record["summary"], record["rows"]
            if context.rank == 0:
                with SummaryWriter(str(arguments.root / "runs" / f"final-{label}-{mode}")) as writer:
                    writer.add_scalar("test-report/macro_spearman", record["summary"]["macro_spearman"], 0)
                print(json.dumps({"event": "final_checkpoint_evaluated", "label": label, "mode": mode,
                                  "macro_spearman": record["summary"]["macro_spearman"]}), flush=True)

    def report():
        paired = {mode: {"A_minus_parent": time_block_difference(rows["parent-328"], rows["A"]),
                         "B_minus_parent": time_block_difference(rows["parent-328"], rows["B"]),
                         "B_minus_A": time_block_difference(rows["A"], rows["B"])} for mode, rows in all_rows.items()}
        primary = {label: value["macro_spearman"] for label, value in summaries["hidden"].items()}
        differences = {"B_minus_parent": primary["B"] - primary["parent-328"], "B_minus_A": primary["B"] - primary["A"]}
        decision = ("revisit_design" if differences["B_minus_parent"] < 0 else
                    "redesign_starting_point" if differences["B_minus_A"] < 0 else "eligible_after_results_review")
        result = {"selection": selection, "registry": registry, "summaries": summaries, "paired_time_blocks": paired,
                  "split": TEST_REPORT_SPLIT, "primary_mode": "hidden", "secondary_mode": "ids_on",
                  "test_used_for_selection": False, "differences": differences,
                  "inconclusive_differences": [name for name, value in differences.items() if abs(value) < 0.004],
                  "experiment193_gate": decision, "experiment193_started": False, "time": time.time()}
        write_json(output / "results.json", result)
        lines = ["# Experiment 1.9.2 Results", "", "## Test-Report (2023-2024)", "",
                 "Both arms were selected using five-score smoothed hidden-ID validation only. Test access followed frozen selection.",
                 "The post-2024 final holdout remains sealed. IDs-on is a secondary inference option.",
                 "", "| Checkpoint | IDs-hidden Spearman | IDs-on Spearman |", "|---|---:|---:|"]
        for label in ("parent-328", "A", "B"):
            lines.append(f"| {label} | {summaries['hidden'][label]['macro_spearman']:.6f} | {summaries['ids_on'][label]['macro_spearman']:.6f} |")
        lines += ["", f"Experiment 1.9.3 gate: {decision}. No subsequent experiment has been started."]
        lines += ["", "Differences below 0.004 are inconclusive. The dataset is survivors-only; historical membership and delisting bias remain.",
                  "The stock holdout excludes only new 1.9.2 exposure, not parent exposure. A flat validation curve alone does not establish a causal ceiling.",
                  "", "Per-market paired 90-session-block intervals and complete aggregate metrics are in final-test/results.json.",
                  "The private cohort definitions and per-basket rows are excluded from public backups."]
        (arguments.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
        write_json(arguments.root / "EXPERIMENT_DONE.json", {"training_complete": True, "final_test_complete": True,
               "final_test_opened": True, "final_holdout_sealed": True,
               "results_sha256": digest(output / "results.json"), "time": time.time()})
    rank_zero(context, report)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "panel", "parent", "prepared"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--end-at", type=float, required=True)
    parser.add_argument("--final-evaluation", action="store_true")
    parser.add_argument("--arm-a-root", type=Path)
    main(parser.parse_args())