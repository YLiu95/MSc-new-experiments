import argparse
import json
import os
from pathlib import Path
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


def main(arguments):
    if not arguments.final_evaluation:
        raise ValueError("Final test is sealed; pass --final-evaluation after both arms finish")
    if os.environ.get("PBS_JOBID") != JOB:
        raise ValueError("Use the retained allocation")
    local_rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, set(range(local_rank * 24, (local_rank + 1) * 24)))
    torch.set_num_threads(4)
    context = ParallelContext.initialize(1)
    if context.dp_size != WORLD * 2:
        raise ValueError(f"Final evaluation requires all {WORLD * 2} allocated GPU workers")
    protocol = rank_zero(context, lambda: load_protocol(arguments.prepared))
    selection = rank_zero(context, lambda: require_final_selection(arguments.root, arguments.final_evaluation, protocol))
    enough_time = rank_zero(context, lambda: time.time() + 3600 < arguments.end_at - 120)
    if not enough_time:
        rank_zero(context, lambda: write_json(arguments.root / "EXPERIMENT_DONE.json", {
            "training_complete": True, "final_test_complete": False, "final_test_opened": False,
            "reason": "insufficient_time_for_full_final_test", "selection": selection, "time": time.time()}))
        dist.barrier()
        dist.destroy_process_group()
        return
    config, _ = rank_zero(context, lambda: verify_parent(arguments.parent))
    panels = [MarketPanel(arguments.panel, market) for market in MARKETS]
    output = arguments.root / "final-test"

    def open_test():
        output.mkdir(mode=0o700, exist_ok=False)
        write_json(output / "OPENED.json", {"selection": selection, "time": time.time(),
                   "parent_manifest_sha256": digest(arguments.parent / "manifest.json"),
                   "user_confirmed_no_previous_test_evaluation": True})
        registry = build_manifest(output / "private-manifest", panels, "test", 9331)
        return registry

    registry = rank_zero(context, open_test)
    items = [json.loads(line) for line in (output / "private-manifest" / registry["file"]).read_text().splitlines()]
    if any(str(panels[item["market"]].dates[item["cutoff"]]) > "2026-08-25" for item in items):
        raise ValueError("Final test exceeds the planned end date")
    model = IdentityHiddenModel(ModelConfig(**config["model"]), context)
    model.requires_grad_(False)
    model.eval()
    summaries, all_rows = {}, {}
    for label, directory in [("parent-328", arguments.parent)] + [(arm, Path(selection[arm]["directory"])) for arm in ("A", "B")]:
        rank_zero(context, lambda: verify(directory, ("weights-00.safetensors", "config.json", "state.json")))
        model.load_state_dict(load_file(directory / "weights-00.safetensors"))
        summary, rows = evaluate_cohort(model, panels, items, registry["tasks"], True)
        summaries[label], all_rows[label] = summary, rows
        rank_zero(context, lambda: write_json(output / f"private-rows-{label}.json", rows))
        if context.rank == 0:
            with SummaryWriter(str(arguments.root / "runs" / f"final-{label}")) as writer:
                writer.add_scalar("test/macro_spearman", summary["macro_spearman"], 0)
            print(json.dumps({"event": "final_checkpoint_evaluated", "label": label, "macro_spearman": summary["macro_spearman"]}), flush=True)

    def report():
        paired = {arm: time_block_difference(all_rows["parent-328"], all_rows[arm]) for arm in ("A", "B")}
        paired["B_minus_A"] = time_block_difference(all_rows["A"], all_rows["B"])
        result = {"selection": selection, "registry": registry, "summaries": summaries, "paired_time_blocks": paired,
                  "test_used_for_selection": False, "identity_masked": True, "time": time.time()}
        write_json(output / "results.json", result)
        write_json(arguments.root / "EXPERIMENT_DONE.json", {"training_complete": True, "final_test_complete": True,
                   "final_test_opened": True, "results_sha256": digest(output / "results.json"), "time": time.time()})
        lines = ["# Experiment 1.9.2 Results", "", "## Final Test", "",
                 "Both arms were selected using five-score smoothed hidden-ID validation only. Test access followed frozen selection.",
                 "", "| Checkpoint | Hidden-ID test Spearman |", "|---|---:|"]
        for label, summary in summaries.items():
            lines.append(f"| {label} | {summary['macro_spearman']:.6f} |")
        lines += ["", "Differences below 0.004 are inconclusive. The dataset is survivors-only; historical membership and delisting bias remain.",
                  "The stock holdout excludes only new 1.9.2 exposure, not parent exposure. A flat validation curve alone does not establish a causal ceiling.",
                  "", "Per-market paired 90-session-block intervals and complete aggregate metrics are in final-test/results.json.",
                  "The private cohort definitions and per-basket rows are excluded from public backups."]
        (arguments.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    rank_zero(context, report)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "panel", "parent", "prepared"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--end-at", type=float, required=True)
    parser.add_argument("--final-evaluation", action="store_true")
    main(parser.parse_args())