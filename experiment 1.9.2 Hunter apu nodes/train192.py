import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import time
import uuid

import numpy as np
from safetensors.torch import load_file
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from ascgpu.model import ParallelContext, clip_global_norm
from ranker.contract import MARKETS
from ranker.data import MarketPanel, write_json
from ranker.losses import pairwise_logistic
from ranker.sampler import Sampler
from ranker.train import informative_count, optimizer_for
from ranking17.model import ModelConfig
from ranking17.recovery import digest
from ranking17.train import batch_for, memory_record
from train19 import checkpoint
from train191 import PARENT_MANIFEST, PARENT_REVISION, distribute_assignments, verify_parent
from evaluate192 import evaluate_cohort
from model192 import IdentityHiddenModel
from policy192 import GLOBAL_BATCH, JOB, MICROBATCH, PARENT_STEP, UPDATE_BUDGET, WORLD, SmoothedSelection, learning_rate
from prepare192 import load_protocol
from sampler192 import FilteredPanel, Sampler192


STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def update(model, wrapped, optimizer, batches, update_index):
    context = model.context
    count = torch.tensor(sum(informative_count(batch) for batch in batches), device=context.device)
    dist.all_reduce(count, group=context.dp_group)
    if not count.item():
        raise RuntimeError("No informative training labels")
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(update_index)
    for group in optimizer.param_groups:
        group["lr"] = rate
    totals = torch.zeros(3, device=context.device, dtype=torch.float64)
    for index, batch in enumerate(batches):
        with wrapped.no_sync() if index + 1 < len(batches) else nullcontext():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = wrapped(batch["inputs"], batch["ticker_ids"], batch["market"], batch["task"], identity_masked=True)[0]
                loss = pairwise_logistic(scores, batch["labels"], batch["observed"], batch["weight"])
                scaled = loss.total * (context.dp_size / count.item())
            scaled.backward()
        totals += torch.stack((loss.total.detach(), loss.pairs.detach(), scores.detach().var(unbiased=False))).double()
    norm = clip_global_norm(model, 1.0)
    dist.all_reduce(totals, group=context.dp_group)
    if not torch.isfinite(totals).all() or not math.isfinite(norm):
        raise RuntimeError("Nonfinite training update")
    optimizer.step()
    torch.cuda.synchronize()
    return {"loss": totals[0].item() / count.item(), "pairs": totals[1].item(), "lr": rate,
            "gradient_norm": norm, "score_variance": totals[2].item() / (len(batches) * context.dp_size)}


def load_parent(parent, model, optimizer):
    recovery = torch.load(parent / "recovery-00.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(recovery["model"])
    optimizer.state.clear()
    optimizer.load_state_dict(recovery["optimizer"])
    del recovery
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(0)
        if group["weight_decay"] not in (0, 0.1):
            raise ValueError("Unexpected parent weight decay")
    seed = model.config.seed + PARENT_STEP * WORLD + model.context.dp_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def rank_zero(context, operation):
    message = [None]
    if context.rank == 0:
        try:
            message[0] = {"value": operation()}
        except Exception as error:
            message[0] = {"error": str(error)}
    dist.broadcast_object_list(message, src=0)
    if "error" in message[0]:
        raise RuntimeError(message[0]["error"])
    return message[0]["value"]


def main(arguments):
    root, panel, parent = arguments.root, arguments.panel, arguments.parent
    stop_at = arguments.end_at - 120
    if os.environ.get("PBS_JOBID") != JOB or time.time() + 900 >= stop_at:
        raise ValueError("The retained allocation with sufficient checkpoint reserve is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    cpus = set(range(local_rank * 24, (local_rank + 1) * 24))
    if not cpus <= os.sched_getaffinity(0):
        raise RuntimeError("Expected 24 physical CPU cores per GPU worker")
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(4)
    context = ParallelContext.initialize(1)
    if context.dp_size != WORLD or torch.cuda.device_count() != 4:
        raise RuntimeError("Each arm requires six nodes with four GPU workers each")
    probe = torch.tensor(context.rank + 1, device=context.device)
    dist.all_reduce(probe)
    if probe.item() != WORLD * (WORLD + 1) // 2:
        raise RuntimeError("24-worker RCCL collective failed")
    torch.use_deterministic_algorithms(True)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    previous, inherited = rank_zero(context, lambda: verify_parent(parent))
    protocol = rank_zero(context, lambda: load_protocol(arguments.prepared))
    for name, key in (("meta.json", "data_meta_sha256"), ("vocabulary.json", "vocabulary_sha256")):
        if digest(panel / name) != previous[key]:
            raise ValueError("Pinned data changed")
    config = {**previous, "experiment": "1.9.2", "arm": arguments.arm, "dp": WORLD,
              "global_batch": GLOBAL_BATCH, "microbatch": MICROBATCH, "new_update_budget": UPDATE_BUDGET,
              "learning_rate": {"kind": "cosine_no_warmup", "start": learning_rate(0), "end": learning_rate(999)},
              "identity_masked_training": True, "market_sampling": "uniform" if arguments.arm == "A" else "eligible_stock_days",
              "allocation_end_epoch": arguments.end_at, "training_stop_epoch": stop_at,
              "protocol_sha256": digest(arguments.prepared / "protocol.json"), "source_sha256": digest(Path(__file__)),
              "migration": {"parent_step": PARENT_STEP, "parent_world": 8, "parent_global_batch": 320,
                            "parent_manifest_sha256": PARENT_MANIFEST, "parent_cloud_revision": PARENT_REVISION,
                            "optimizer_preserved": True, "parent_counters_preserved": True,
                            "filtered_counters_new": True, "exact_parent_baskets_excluded": True,
                            "rng_reseeded": True, "bitwise_continuation": False}}
    state = {**inherited, "world": WORLD, "experiment": "1.9.2", "arm": arguments.arm, "new_updates": 0}
    model_config = ModelConfig(**config["model"])
    model = IdentityHiddenModel(model_config, context)
    if model.parameter_count() != 1069302417 or model_config.dropout != 0.15:
        raise ValueError("Parent model contract changed")
    model.load_state_dict(load_file(parent / "weights-00.safetensors"))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], process_group=context.dp_group,
                                      broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=128)
    optimizer = optimizer_for(model, context.device)
    synthetic = {"inputs": torch.linspace(-1, 1, MICROBATCH * 128 * 512, device=context.device).reshape(MICROBATCH, 128, 512),
                 "ticker_ids": torch.arange(128, device=context.device)[None].repeat(MICROBATCH, 1),
                 "market": torch.zeros(MICROBATCH, dtype=torch.long, device=context.device),
                 "task": torch.tensor([[512, 90, 128]], device=context.device).repeat(MICROBATCH, 1),
                 "labels": torch.arange(128, device=context.device).float()[None].repeat(MICROBATCH, 1),
                 "observed": torch.ones(MICROBATCH, 128, dtype=torch.bool, device=context.device),
                 "weight": torch.ones(MICROBATCH, device=context.device)}
    started = time.monotonic()
    update(model, wrapped, optimizer, [synthetic], 0)
    gate_seconds = time.monotonic() - started
    memory = memory_record(context, "24_gpu_hidden_worst_microbatch_backward_adam")
    if any(record["peak_reserved_bytes"] > 0.9 * record["hip_total_bytes"] for record in memory):
        raise RuntimeError("Worst-case microbatch exceeds the 90 percent memory gate")
    del synthetic
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    torch.cuda.empty_cache()
    load_parent(parent, model, optimizer)
    torch.cuda.empty_cache()
    originals = [MarketPanel(panel, market) for market in MARKETS]
    panels = [FilteredPanel(panel, market, arguments.prepared) for market in MARKETS]
    cohorts = json.loads((arguments.prepared / "private-cohorts.json").read_text())
    sampler, writer = None, None

    def initialize():
        nonlocal sampler, writer
        if (root / "latest.json").exists():
            raise ValueError("Run already contains a checkpoint")
        sampler = Sampler192.from_parent(panels, originals, parent / "sampler.sqlite", root / "ledgers" / f"{uuid.uuid4().hex}.sqlite",
                                        arguments.arm, protocol["market_weights"], model_config.seed)
        reference = Sampler(originals, root / "ledgers" / "catalogue-check.sqlite", model_config.seed)
        try:
            if sampler.catalogue != reference.catalogue or sampler.issued != inherited["scheduled"]:
                raise ValueError("Filtering changed the parent task catalogue or scheduled progress")
        finally:
            reference.close()
        writer = SummaryWriter(str(root / "runs" / f"arm-{arguments.arm}"))
        write_json(root / "config.json", config)
        write_json(root / "reports/preflight.json", {"passed": True, "world": WORLD, "global_batch": GLOBAL_BATCH,
                   "microbatch": MICROBATCH, "worst_microbatch_seconds": gate_seconds, "memory_phases": memory,
                   "synthetic_update_rolled_back": True, "parent_full_payload_verified": True})
        write_json(root / "reports/protocol.json", protocol)
        writer.add_scalar("preflight/workers", WORLD, 0)
        writer.flush()
        return True

    rank_zero(context, initialize)
    print(json.dumps({"event": "worker_ready", "arm": arguments.arm, "rank": context.rank,
                      "host": socket.gethostname(), "pid": os.getpid(), "cpus": sorted(os.sched_getaffinity(0))}), flush=True)

    def evaluate(include_stock_cohorts):
        result = {}
        conditions = [("validation", "validation", True), ("validation_ids_on", "validation", False)]
        if include_stock_cohorts:
            conditions += [("included", "included", True), ("held_out", "held_out", True)]
        for name, cohort, masked in conditions:
            summary, rows = evaluate_cohort(model, originals, cohorts[cohort], protocol["tasks"], masked)
            result[name] = summary
            def record():
                write_json(root / "private-evaluation" / f"update-{state['new_updates']:04d}-{name}.json", rows)
                writer.add_scalar(f"{name}/macro_spearman", summary["macro_spearman"], state["new_updates"])
                for market, value in summary["by_market"].items():
                    writer.add_scalar(f"{name}/market/{market}", value, state["new_updates"])
                return True
            rank_zero(context, record)
        return result

    started = time.monotonic()
    baseline = evaluate(True)
    validation_seconds = max(180, (time.monotonic() - started) * 1.5)
    selection = SmoothedSelection(baseline["validation"]["macro_spearman"])
    state["best_metric"] = selection.best_smoothed
    state["selection"] = selection.state()
    rank_zero(context, lambda: write_json(root / "reports/baseline.json", baseline))
    started = time.monotonic()
    checkpoint(root, model, optimizer, state, sampler, config, best=True)
    save_seconds = max(120, (time.monotonic() - started) * 1.5 + 15)
    saved_step, validated_update, update_seconds = state["step"], 0, 30
    early_stop, stop_reason = False, None

    def materialize(entries):
        batch = batch_for(panels, entries, torch.device("cpu"), model_config.tickers)
        batch["weight"] = torch.ones(len(entries))
        return {name: value.pin_memory() for name, value in batch.items()}

    with ThreadPoolExecutor(max_workers=4) as workers:
        while True:
            stopping = torch.tensor(int(STOP), device=context.device)
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)

            def decide():
                if stopping.item():
                    return "finish", "signal"
                if time.time() + save_seconds + update_seconds >= stop_at:
                    return "finish", "allocation_reserve"
                if early_stop:
                    return "finish", "early_stop"
                due = state["new_updates"] > validated_update and state["new_updates"] % 50 == 0
                if due:
                    if time.time() + validation_seconds + 2 * save_seconds + update_seconds >= stop_at:
                        return "finish", "validation_would_exceed_reserve"
                    return "validate", None
                if state["new_updates"] >= UPDATE_BUDGET:
                    return "finish", "update_budget"
                return "update", None

            action, stop_reason = rank_zero(context, decide)
            if action in ("validate", "finish"):
                improved = False
                if action == "validate":
                    started = time.monotonic()
                    summaries = evaluate(state["new_updates"] % 200 == 0)
                    validation_seconds = max(validation_seconds, 1.5 * (time.monotonic() - started))
                    decision = rank_zero(context, lambda: selection.observe(summaries["validation"]["macro_spearman"]))
                    state["selection"] = rank_zero(context, selection.state)
                    improved, early_stop = decision["best"], decision["stop"]
                    state["best_metric"] = state["selection"]["best_smoothed"]
                    def record_validation():
                        record = {"step": state["step"], "new_updates": state["new_updates"], "selection": decision,
                                  "results": summaries, "time": time.time()}
                        with (root / "validation.jsonl").open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        if decision["smoothed"] is not None:
                            writer.add_scalar("validation/smoothed_macro_spearman", decision["smoothed"], state["new_updates"])
                        writer.add_scalar("selection/stale_validations", decision["stale_validations"], state["new_updates"])
                        if "included" in summaries:
                            writer.add_scalar("holdout/included_minus_held_out", summaries["included"]["macro_spearman"] -
                                              summaries["held_out"]["macro_spearman"], state["new_updates"])
                        writer.flush()
                        print(json.dumps({"event": "validation", "arm": arguments.arm, **record}), flush=True)
                    rank_zero(context, record_validation)
                    validated_update = state["new_updates"]
                if state["step"] != saved_step:
                    started = time.monotonic()
                    checkpoint(root, model, optimizer, state, sampler, config, best=improved)
                    save_seconds = max(save_seconds, (time.monotonic() - started) * 1.5 + 15)
                    saved_step = state["step"]
                if action == "finish":
                    def finish():
                        writer.flush()
                        report = {**state, "reason": stop_reason, "finished_at": time.time(), "checkpoint_complete": True,
                                  "finished_before_reserve": time.time() <= stop_at, "training_stop_epoch": stop_at,
                                  "allocation_end_epoch": arguments.end_at, "selection_frozen": True}
                        write_json(root / "TRAINING_DONE.json", report)
                        print(json.dumps({"event": "training_done", **report}), flush=True)
                    rank_zero(context, finish)
                    break
                continue
            started = time.monotonic()
            source = None
            def schedule():
                nonlocal source
                source = distribute_assignments(sampler.issue(GLOBAL_BATCH), world=WORLD)
                return True
            rank_zero(context, schedule)
            assignments = [None]
            dist.scatter_object_list(assignments, source, src=0)
            cpu_batches = list(workers.map(materialize, assignments[0]))
            batches = [{name: value.to(context.device, non_blocking=True) for name, value in batch.items()} for batch in cpu_batches]
            del cpu_batches
            data_seconds = time.monotonic() - started
            report = update(model, wrapped, optimizer, batches, state["new_updates"])
            del batches
            state["step"] += 1
            state["new_updates"] += 1
            state["scheduled"] += GLOBAL_BATCH
            elapsed = torch.tensor(time.monotonic() - started, device=context.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            update_seconds = max(update_seconds, elapsed.item() * 1.5)
            def record_update():
                exposure = sampler.exposure()
                record = {"step": state["step"], "new_updates": state["new_updates"], "scheduled": state["scheduled"],
                          **report, "seconds": elapsed.item(), "data_seconds": data_seconds,
                          "samples_per_second": GLOBAL_BATCH / elapsed.item(), "global_batch": GLOBAL_BATCH,
                          "exposure": exposure, "time": time.time()}
                with (root / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                for name, value in report.items():
                    writer.add_scalar("train/" + name, value, state["new_updates"])
                for market, values in exposure.items():
                    for name, value in values.items():
                        writer.add_scalar(f"exposure/{market}/{name}", value, state["new_updates"])
                writer.add_scalar("system/samples_per_second", record["samples_per_second"], state["new_updates"])
                writer.add_scalar("system/data_seconds", data_seconds, state["new_updates"])
                writer.flush()
                print(json.dumps({"event": "update", "arm": arguments.arm, **record}), flush=True)
            rank_zero(context, record_update)
    if context.rank == 0:
        sampler.close()
        writer.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--end-at", type=float, required=True)
    main(parser.parse_args())