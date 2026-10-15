import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid

import numpy as np
from safetensors.torch import load_file
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from ascgpu.model import ParallelContext, clip_global_norm
from ranker.contract import MARKETS, REVISION
from ranker.data import MarketPanel, PREPARATION_POLICY, write_json
from ranker.losses import pairwise_logistic
from ranker.sampler import Sampler
from ranker.train import informative_count, learning_rate, optimizer_for
from ranking17.model import ModelConfig, RankingModel
from ranking17.recovery import digest, verify
from ranking17.train import batch_for, memory_record, validate


WORLD = 48
GLOBAL_BATCH = 1536
PARENT_BATCH = 320
MICROBATCH = 2
LR_CAP = 0.00006
PARENT_STEP = 328
PARENT_MANIFEST = "133e851f4e4de23886b173f4c2874c8a9645e2531a185edde618256f607f2b01"
PARENT_REVISION = "7a17855a414de2241469c1e3fd78f5bf351a69dc"
STOP = False


def capped_learning_rate(step):
    return min(learning_rate(step), LR_CAP)


def distribute_assignments(entries, world=WORLD):
    if len(entries) != GLOBAL_BATCH or len(entries) % (world * MICROBATCH):
        raise ValueError("Batch must divide evenly across all GPU workers")
    groups = [entries[offset:offset + MICROBATCH] for offset in range(0, len(entries), MICROBATCH)]
    if any(len({entry.task for entry in group}) != 1 for group in groups):
        raise ValueError("Microbatches must contain matching task shapes")
    groups.sort(key=lambda group: group[0].task[0] * group[0].task[2], reverse=True)
    assigned, loads = [[] for _ in range(world)], [0] * world
    for offset in range(0, len(groups), world):
        order = sorted(range(world), key=lambda rank: (loads[rank], rank))
        for rank, group in zip(order, groups[offset:offset + world]):
            assigned[rank].append(group)
            loads[rank] += group[0].task[0] * group[0].task[2]
    return assigned


def next_action(now, validation_at, stop_at, save_seconds, validation_seconds=120, update_seconds=10):
    if now + save_seconds + update_seconds >= stop_at:
        return "finish"
    if now >= validation_at and now + validation_seconds + 2 * save_seconds + update_seconds < stop_at:
        return "validate"
    return "update"


def migrated_state(config, previous):
    if config["tp"] != 1 or config["dp"] != 8 or previous["world"] != 8:
        raise ValueError("Expected the eight-worker Experiment 1.9 checkpoint")
    if previous["step"] != PARENT_STEP or previous["scheduled"] != PARENT_STEP * PARENT_BATCH:
        raise ValueError("Experiment 1.9.1 must start from step 328")
    if config["global_batch"] != PARENT_BATCH or config["parameters"] != 1069302417:
        raise ValueError("Parent architecture or batch contract changed")
    return {**previous, "world": WORLD, "experiment": "1.9.1", "migration_parent_world": 8,
            "migration_parent_step": PARENT_STEP, "migration_rng_reseeded": True,
            "migration_bitwise_continuation": False}


def verify_parent(directory):
    if digest(directory / "manifest.json") != PARENT_MANIFEST:
        raise ValueError("Parent differs from the pinned private cloud checkpoint")
    verify(directory, workers=4)
    config = json.loads((directory / "config.json").read_text())
    previous = json.loads((directory / "state.json").read_text())
    return config, migrated_state(config, previous)


def load_parent(directory, model, optimizer):
    recovery = torch.load(directory / "recovery-00.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(recovery["model"])
    optimizer.state.clear()
    optimizer.load_state_dict(recovery["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = capped_learning_rate(PARENT_STEP)
    seed = model.config.seed + PARENT_STEP * WORLD + model.context.dp_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def update(model, wrapped, optimizer, batches, step):
    context = model.context
    count = torch.tensor(sum(informative_count(batch) for batch in batches), device=context.device)
    dist.all_reduce(count, group=context.dp_group)
    if not count.item():
        raise RuntimeError("No informative labels in scheduled batch")
    optimizer.zero_grad(set_to_none=True)
    rate = capped_learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = rate
    totals = torch.zeros(3, device=context.device, dtype=torch.float64)
    for index, batch in enumerate(batches):
        synchronization = wrapped.no_sync() if index + 1 < len(batches) else nullcontext()
        with synchronization:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = wrapped(batch["inputs"], batch["ticker_ids"], batch["market"], batch["task"])[0]
                loss = pairwise_logistic(scores, batch["labels"], batch["observed"], batch["weight"])
                scaled = loss.total * (context.dp_size / count.item())
            scaled.backward()
        totals += torch.stack((loss.total.detach(), loss.pairs.detach(), scores.detach().var(unbiased=False))).double()
    norm = clip_global_norm(model, 1.0)
    dist.all_reduce(totals, group=context.dp_group)
    if not torch.isfinite(totals).all() or not math.isfinite(norm):
        raise RuntimeError("Nonfinite update")
    optimizer.step()
    torch.cuda.synchronize()
    return {"loss": totals[0].item() / count.item(), "pairs": totals[1].item(), "lr": rate,
            "gradient_norm": norm, "score_variance": totals[2].item() / (len(batches) * context.dp_size)}


def request_stop(signum, frame):
    global STOP
    STOP = True


def main(arguments):
    from train19 import checkpoint

    root, panel, parent = arguments.root, arguments.panel, arguments.parent
    stop_at = arguments.end_at - 120
    if time.time() + 600 >= stop_at:
        raise ValueError("Insufficient allocation time for startup and final recovery")
    local_rank = int(os.environ["LOCAL_RANK"])
    cpus = set(range(local_rank * 24, (local_rank + 1) * 24))
    if not cpus <= os.sched_getaffinity(0):
        raise RuntimeError("Expected 24 physical CPU cores per GPU worker")
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(4)
    context = ParallelContext.initialize(1)
    if context.dp_size != WORLD or torch.cuda.device_count() != 4:
        raise RuntimeError("Experiment 1.9.1 requires twelve nodes with four GPUs each")
    probe = torch.tensor(context.rank + 1, device=context.device)
    dist.all_reduce(probe)
    if probe.item() != WORLD * (WORLD + 1) // 2:
        raise RuntimeError("48-worker RCCL collective failed")
    torch.use_deterministic_algorithms(True)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    prepared = [None]
    if context.rank == 0:
        try:
            previous, state = verify_parent(parent)
            prepared[0] = {"config": previous, "state": state}
        except Exception as error:
            prepared[0] = {"error": str(error)}
    dist.broadcast_object_list(prepared, src=0)
    if "error" in prepared[0]:
        raise RuntimeError(prepared[0]["error"])
    previous, state = prepared[0]["config"], prepared[0]["state"]
    config = {**previous, "experiment": "1.9.1", "dp": WORLD, "global_batch": GLOBAL_BATCH,
              "microbatch": MICROBATCH, "learning_rate_cap": LR_CAP, "allocation_end_epoch": arguments.end_at,
              "training_stop_epoch": stop_at, "cutoff_policy": "finish checkpoint before allocation end minus 120 seconds",
              "source_sha256": digest(Path(__file__)), "rng_reset": True,
              "migration": {"source": str(parent), "parent_step": PARENT_STEP, "parent_world": 8,
                            "parent_manifest_sha256": PARENT_MANIFEST, "parent_cloud_revision": PARENT_REVISION,
                            "optimizer_preserved": True, "sampler_preserved": True, "rng_reseeded": True,
                            "parent_global_batch": PARENT_BATCH, "bitwise_continuation": False}}
    for name, key in (("meta.json", "data_meta_sha256"), ("vocabulary.json", "vocabulary_sha256")):
        if digest(panel / name) != config[key]:
            raise ValueError("Pinned data fingerprint changed")
    metadata = json.loads((panel / "meta.json").read_text())
    if metadata["revision"] != REVISION or metadata["preparation_policy"] != PREPARATION_POLICY:
        raise ValueError("Corrected pinned panels required")
    model_config = ModelConfig(**config["model"])
    model = RankingModel(model_config, context)
    if model.parameter_count() != config["parameters"]:
        raise ValueError("Parent architecture mismatch")
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
    update(model, wrapped, optimizer, [synthetic], PARENT_STEP)
    gate_seconds = time.monotonic() - started
    memory = memory_record(context, "48_gpu_worst_microbatch_backward_adam")
    if any(record["peak_reserved_bytes"] > 0.9 * record["hip_total_bytes"] for record in memory):
        raise RuntimeError("Worst-case microbatch exceeds the 90 percent memory gate")
    del synthetic
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    torch.cuda.empty_cache()
    load_parent(parent, model, optimizer)
    torch.cuda.empty_cache()
    panels = [MarketPanel(panel, market) for market in MARKETS]
    registry = json.loads((panel / "evaluation" / "val-primary.json").read_text())
    if digest(panel / "evaluation" / registry["file"]) != config["validation_sha256"]:
        raise ValueError("Parent validation registry changed")
    sampler, writer = None, None
    restored = [None]
    if context.rank == 0:
        try:
            sampler = Sampler.restore(panels, parent / "sampler.sqlite", root / "ledgers" / f"{uuid.uuid4().hex}.sqlite", model_config.seed)
            if sampler.issued != state["scheduled"]:
                raise ValueError("Sampler progress differs from the parent checkpoint")
            writer = SummaryWriter(str(root / "runs"))
            write_json(root / "config.json", config)
            report = {"passed": True, "world": WORLD, "tp": 1, "dp": WORLD, "parameters": model.parameter_count(),
                      "microbatch": MICROBATCH, "global_batch": GLOBAL_BATCH, "learning_rate_cap": LR_CAP,
                      "worst_microbatch_seconds": gate_seconds, "memory_phases": memory,
                      "synthetic_update_rolled_back": True, "resume_bitwise_tested": False}
            write_json(root / "reports" / "preflight.json", report)
            write_json(root / "reports" / "migration.json", {**config["migration"], **state})
            writer.add_scalar("preflight/worst_microbatch_seconds", gate_seconds, PARENT_STEP)
            writer.add_scalar("preflight/workers", WORLD, PARENT_STEP)
            writer.add_scalar("validation/macro_spearman", state["best_metric"], PARENT_STEP)
            writer.flush()
            print(json.dumps({"event": "migration_restored", **state, "lr_cap": LR_CAP, "stop_at": stop_at}), flush=True)
        except Exception as error:
            restored[0] = str(error)
    dist.broadcast_object_list(restored, src=0)
    if restored[0] is not None:
        raise RuntimeError(restored[0])
    started = time.monotonic()
    checkpoint(root, model, optimizer, state, sampler, config, best=True)
    save_seconds = max(120, (time.monotonic() - started) * 1.5 + 15)
    saved_step = state["step"]
    validation_at, validation_seconds, update_seconds = time.time() + 300, 120, 10

    def materialize(entries):
        batch = batch_for(panels, entries, torch.device("cpu"), model_config.tickers)
        batch["weight"] = torch.ones(len(entries))
        return {name: value.pin_memory() for name, value in batch.items()}

    with ThreadPoolExecutor(max_workers=4) as workers:
        while True:
            stopping = torch.tensor(int(STOP), device=context.device)
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            decision = [None]
            if context.rank == 0:
                decision[0] = "finish" if stopping.item() else next_action(
                    time.time(), validation_at, stop_at, save_seconds, validation_seconds, update_seconds)
            dist.broadcast_object_list(decision, src=0)
            action = decision[0]
            if action in ("validate", "finish"):
                improved = False
                if action == "validate":
                    started = time.monotonic()
                    summary = validate(model, panels, panel, registry)
                    validation_seconds = max(validation_seconds, 1.5 * (time.monotonic() - started))
                    metric = summary["macro_spearman"]
                    improved = metric is not None and math.isfinite(metric) and metric > state["best_metric"]
                    if improved:
                        state["best_metric"] = metric
                    if context.rank == 0:
                        with (root / "validation.jsonl").open("a") as stream:
                            stream.write(json.dumps({"step": state["step"], "monitoring_subset": True, **summary}) + "\n")
                        if metric is not None:
                            writer.add_scalar("validation/macro_spearman", metric, state["step"])
                        print(json.dumps({"event": "validation", "step": state["step"], "macro_spearman": metric}), flush=True)
                    validation_at = time.time() + 300
                if state["step"] != saved_step:
                    started = time.monotonic()
                    checkpoint(root, model, optimizer, state, sampler, config, best=improved)
                    save_seconds = max(save_seconds, (time.monotonic() - started) * 1.5 + 15)
                    saved_step = state["step"]
                if action == "finish":
                    if context.rank == 0:
                        report = {**state, "finished_at": time.time(), "training_stop_epoch": stop_at,
                                  "allocation_end_epoch": arguments.end_at, "checkpoint_complete": True,
                                  "finished_before_reserve": time.time() <= stop_at}
                        write_json(root / "TRAINING_DONE.json", report)
                        print(json.dumps({"event": "training_done", **report}), flush=True)
                    break
                continue
            started = time.monotonic()
            assignments = [None]
            source = distribute_assignments(sampler.issue(GLOBAL_BATCH)) if context.rank == 0 else None
            dist.scatter_object_list(assignments, source, src=0)
            cpu_batches = list(workers.map(materialize, assignments[0]))
            batches = [{name: value.to(context.device, non_blocking=True) for name, value in batch.items()}
                       for batch in cpu_batches]
            del cpu_batches
            data_seconds = time.monotonic() - started
            report = update(model, wrapped, optimizer, batches, state["step"])
            del batches
            state["step"] += 1
            state["scheduled"] += GLOBAL_BATCH
            elapsed = torch.tensor(time.monotonic() - started, device=context.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            update_seconds = max(update_seconds, elapsed.item() * 1.5)
            if context.rank == 0:
                record = {**state, **report, "seconds": elapsed.item(), "data_seconds": data_seconds,
                          "samples_per_second": GLOBAL_BATCH / elapsed.item(), "global_batch": GLOBAL_BATCH}
                with (root / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                for name, value in report.items():
                    writer.add_scalar("train/" + name, value, state["step"])
                writer.add_scalar("system/samples_per_second", record["samples_per_second"], state["step"])
                writer.add_scalar("system/data_seconds", data_seconds, state["step"])
                writer.flush()
                print(json.dumps({"event": "update", **record}), flush=True)
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
    parser.add_argument("--end-at", type=float, required=True)
    main(parser.parse_args())