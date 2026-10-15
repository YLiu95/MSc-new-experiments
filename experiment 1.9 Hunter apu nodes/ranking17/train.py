import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from ascgpu.model import ParallelContext, clip_global_norm
from ranker.contract import MARKETS, REVISION, Task
from ranker.data import MarketPanel, PREPARATION_POLICY, write_json
from ranker.evaluate import basket_metrics, macro_spearman
from ranker.losses import pairwise_logistic
from ranker.sampler import Sampler
from ranker.train import informative_count, learning_rate, optimizer_for
from .model import ModelConfig, RankingModel
from .offline import maintain
from .recovery import backup, digest, restore, restore_migrated, save, tensor_digest
from .warmstart import load_warm_start


STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def batch_for(panels, entries, device, unknown):
    samples = [panels[entry.market].sample(entry.cutoff, Task(*entry.task), entry.tickers, unknown) for entry in entries]
    return {key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=device)
            for key in ("inputs", "ticker_ids", "market", "task", "labels", "observed")}


def update(model, wrapped, optimizer, batches, step):
    context = model.context
    count = torch.tensor(sum(informative_count(batch) for batch in batches), device=context.device)
    dist.all_reduce(count, group=context.dp_group)
    if not count.item():
        raise RuntimeError("No informative labels in scheduled batch")
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = rate
    totals = torch.zeros(3, device=context.device, dtype=torch.float64)
    for index, batch in enumerate(batches):
        synchronization = wrapped.no_sync() if context.dp_size > 1 and index + 1 < len(batches) else nullcontext()
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


def validate(model, panels, panel, registry):
    context = model.context
    model.eval()
    allowed = set(registry["monitor_line_indices"])
    lines = (panel / "evaluation" / registry["file"]).read_text().splitlines()
    selected = [json.loads(line) for index, line in enumerate(lines) if index in allowed]
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, item in enumerate(selected):
            if index % context.dp_size != context.dp_rank:
                continue
            sample = panels[item["market"]].sample(item["cutoff"], Task(*item["task"]), item["tickers"], model.config.tickers)
            scores = model(torch.as_tensor(sample["inputs"], device=context.device)[None],
                           torch.as_tensor(sample["ticker_ids"], device=context.device)[None],
                           torch.tensor([sample["market"]], device=context.device),
                           torch.tensor([sample["task"]], device=context.device))[0][0].float().cpu().numpy()
            rows.append({**basket_metrics(scores, sample["labels"]), "task": sample["task"],
                         "market": MARKETS[item["market"]], "cutoff": item["cutoff"]})
    received = [None] * context.dp_size
    dist.all_gather_object(received, rows, group=context.dp_group)
    rows = [row for chunk in received for row in chunk]
    summary = macro_spearman(rows, tuple(map(tuple, registry["tasks"])))
    cells = defaultdict(list)
    for row in rows:
        if row["spearman"] is not None:
            cells[f"{row['task']}/{row['market']}"].append(row["spearman"])
    summary["market_tasks"] = {key: float(np.mean(values)) for key, values in cells.items()}
    model.train()
    return summary


def memory_record(context, phase):
    import resource
    free, total = torch.cuda.mem_get_info(context.device)
    record = {"phase": phase, "rank": context.rank, "local_rank": context.local_rank,
              "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "hip_free_bytes": free, "hip_total_bytes": total,
              "allocated_bytes": torch.cuda.memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "host_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    host = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    record["host_mem_available_bytes"] = int(host["MemAvailable"].split()[0]) * 1024
    received = [None] * dist.get_world_size()
    dist.all_gather_object(received, record)
    if context.rank == 0:
        print(json.dumps({"event": "memory", "records": received}), flush=True)
    return received


def maintain_distributed(root, context):
    result = [None]
    if context.rank == 0:
        try:
            result[0] = {"seconds": maintain(root)}
        except Exception as error:
            result[0] = {"error": str(error)}
    dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["seconds"]


def run(arguments):
    root = arguments.root
    config = json.loads((root / "config.json").read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    cpus = set(range(local_rank * 24, (local_rank + 1) * 24))
    if not cpus <= os.sched_getaffinity(0):
        raise ValueError("Launcher restricted CPU affinity; expected 24 worker-local cores")
    os.sched_setaffinity(0, cpus)
    context = ParallelContext.initialize(config["tp"])
    if context.tp_size != 4 or context.dp_size != config["dp"]:
        raise ValueError("Hunter requires four APU processes per node and the frozen DP layout")
    probe = torch.tensor(context.rank + 1, device=context.device)
    dist.all_reduce(probe)
    world = dist.get_world_size()
    if probe.item() != world * (world + 1) // 2:
        raise RuntimeError("Inter-node RCCL reduction failed")
    torch.use_deterministic_algorithms(True)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    panel = arguments.panel
    metadata = json.loads((panel / "meta.json").read_text())
    if metadata["revision"] != REVISION or metadata["preparation_policy"] != PREPARATION_POLICY:
        raise ValueError("Corrected pinned panels required")
    if digest(panel / "meta.json") != config["data_meta_sha256"] or digest(panel / "vocabulary.json") != config["vocabulary_sha256"]:
        raise ValueError("Frozen data metadata mismatch")
    model_config = ModelConfig(**config["model"])
    torch.manual_seed(model_config.seed)
    model = RankingModel(model_config, context)
    if model.parameter_count() != config["parameters"] or model.parameter_count() <= 10_000_000_000:
        raise ValueError("Parameter-count gate failed")
    provenance = None
    if not arguments.resume and not arguments.migrate_from:
        if arguments.warm_start is None:
            raise ValueError("The approved step-153 warm start is required")
        if digest(arguments.warm_start / "download.json") != config["warm_start_manifest_sha256"]:
            raise ValueError("Frozen warm-start manifest differs")
        provenance = load_warm_start(model, arguments.warm_start)
        if context.rank == 0:
            write_json(root / "reports" / "warm_start.json", provenance)
    memory_reports = [memory_record(context, "model_loaded")]
    if context.rank == 0:
        print(json.dumps({"event": "model_initialized", "parameters": model.parameter_count(),
                          "tp": context.tp_size, "dp": context.dp_size,
                          "device": torch.cuda.get_device_name(context.device), "preflight": arguments.gate}), flush=True)
    optimizer = optimizer_for(model, context.device)
    wrapped = DistributedDataParallel(model, device_ids=[context.local_rank], process_group=context.dp_group,
                                      broadcast_buffers=False, gradient_as_bucket_view=True) if context.dp_size > 1 else model
    torch.manual_seed(model_config.seed + context.dp_rank)
    panels = [MarketPanel(panel, market) for market in MARKETS]
    registry = json.loads((panel / "evaluation" / "val-primary.json").read_text())
    if digest(panel / "evaluation" / registry["file"]) != config["validation_sha256"]:
        raise ValueError("Frozen validation mismatch")
    ledger = root / "ledgers" / f"{uuid.uuid4().hex}.sqlite"
    sampler = Sampler(panels, ledger, model_config.seed) if context.rank == 0 else None
    state = {"step": config["parent_step"], "scheduled": 0, "best_metric": None,
             "world": dist.get_world_size(), "warm_start_parent_step": config["parent_step"]}
    writer = SummaryWriter(str(root / "runs")) if context.rank == 0 else None
    if arguments.gate:
        gate_root = root / "preflight"
        values = torch.linspace(-1, 1, 128 * 512, device=context.device).reshape(1, 128, 512)
        batch = {"inputs": values, "ticker_ids": torch.arange(128, device=context.device)[None],
                 "market": torch.zeros(1, dtype=torch.long, device=context.device),
                 "task": torch.tensor([[512, 90, 128]], device=context.device),
                 "labels": torch.arange(128, device=context.device).float()[None],
                 "observed": torch.ones(1, 128, dtype=torch.bool, device=context.device),
                 "weight": torch.ones(1, device=context.device)}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        report = update(model, wrapped, optimizer, [batch], 0)
        memory_reports.append(memory_record(context, "backward_and_adam"))
        seconds = time.monotonic() - started
        peak = torch.tensor(torch.cuda.max_memory_reserved(), device=context.device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if peak.item() > torch.cuda.get_device_properties(context.device).total_memory * 0.90:
            raise RuntimeError("Worst-case reserved GPU memory exceeds 90 percent")
        if context.rank == 0:
            print(json.dumps({"event": "preflight_memory", "seconds": seconds,
                              "peak_reserved_gib_all_ranks": peak.item() / 2**30}), flush=True)
        state["step"] = 1
        saved, checkpoint_seconds = save(gate_root, model, optimizer, state, sampler, config)
        memory_reports.append(memory_record(context, "checkpoint_saved"))
        if context.rank == 0:
            print(json.dumps({"event": "preflight_checkpoint", "seconds": checkpoint_seconds}), flush=True)
        copied = arguments.backup / "preflight" / saved.name
        backup_start = time.monotonic()
        if context.rank == 0:
            backup(saved, copied)
        dist.barrier()
        backup_seconds = time.monotonic() - backup_start
        if context.rank == 0:
            print(json.dumps({"event": "preflight_backup_verified", "seconds": backup_seconds}), flush=True)
        next_entries = sampler.issue(4) if context.rank == 0 else None
        update(model, wrapped, optimizer, [batch], 1)
        expected = tensor_digest(model, optimizer)
        restore(copied, model, optimizer, config)
        memory_reports.append(memory_record(context, "checkpoint_restored"))
        update(model, wrapped, optimizer, [batch], 1)
        if tensor_digest(model, optimizer) != expected:
            raise RuntimeError("Resume-and-next-update is not bitwise equivalent")
        peak = torch.tensor(torch.cuda.max_memory_reserved(), device=context.device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if peak.item() > torch.cuda.get_device_properties(context.device).total_memory * 0.90:
            raise RuntimeError(f"Restored-update reserved GPU memory exceeds 90 percent: {peak.item() / 2**30:.2f} GiB")
        if context.rank == 0:
            restored_sampler = Sampler.restore(panels, copied / "sampler.sqlite", ledger.with_name(uuid.uuid4().hex + ".sqlite"), model_config.seed)
            if restored_sampler.issue(4) != next_entries:
                raise RuntimeError("Logical sample sequence changed after restore")
            restored_sampler.close()
            if checkpoint_seconds + backup_seconds > 600:
                raise RuntimeError("Measured recovery/backup duration leaves insufficient shutdown reserve")
            for name, value in {"worst_microbatch_seconds": seconds, "checkpoint_seconds": checkpoint_seconds,
                                "backup_seconds": backup_seconds, "peak_reserved_gib": peak.item() / 2**30}.items():
                writer.add_scalar("preflight/" + name, value, 0)
            writer.flush()
            write_json(root / "reports" / "preflight.json", {"passed": True, "parameters": model.parameter_count(),
                       "source_sha256": config["source_sha256"], "warm_start": provenance,
                       "memory_phases": memory_reports,
                       "world": dist.get_world_size(), "worst_microbatch_seconds": seconds,
                       "peak_reserved_gib_all_ranks": peak.item() / 2**30, "checkpoint_seconds": checkpoint_seconds,
                       "backup_seconds": backup_seconds, "resume_next_update_bitwise_equal": True,
                       "sampler_next_visit_equal": True, "synthetic_loss": report["loss"]})
            print(json.dumps({"event": "preflight_passed", "resume_next_update_bitwise_equal": True,
                              "sampler_next_visit_equal": True}), flush=True)
        dist.barrier()
    else:
        gate = json.loads((root / "reports" / "preflight.json").read_text())
        if not gate["passed"] or gate["world"] != dist.get_world_size():
            raise ValueError("Matching full-model preflight required")
        continuation = arguments.resume or arguments.migrate_from
        if continuation:
            restore_checkpoint = restore_migrated if arguments.migrate_from else restore
            state = restore_checkpoint(continuation, model, optimizer, config)
            if context.rank == 0:
                sampler.close()
                sampler = Sampler.restore(panels, continuation / "sampler.sqlite", ledger.with_name(uuid.uuid4().hex + ".sqlite"), model_config.seed)
                if sampler.issued != state["scheduled"]:
                    raise ValueError("Sampler count differs from completed sample count")
                with (root / "restarts.jsonl").open("a") as stream:
                    stream.write(json.dumps({"from": str(continuation), "time": time.time(), "step": state["step"],
                                             "topology_migration": bool(arguments.migrate_from)}) + "\n")
                if arguments.migrate_from:
                    write_json(root / "reports" / "migration.json", {**config["migration"], **state})
                    print(json.dumps({"event": "migration_restored", **state}), flush=True)
            if arguments.migrate_from:
                save(root, model, optimizer, state, sampler, config)
                maintain_distributed(root, context)
        next_save = time.monotonic() + 1800
        last_saved = state["step"] if continuation else -1
        worst_update = gate["worst_microbatch_seconds"] * math.ceil(320 / context.dp_size) * 1.5
        while True:
            stopping = torch.tensor(int(STOP or (root / "control" / "STOP").exists()
                                        or time.time() + worst_update >= arguments.stop_at), device=context.device)
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                break
            started = time.monotonic()
            issued = [sampler.issue(320) if context.rank == 0 else None]
            dist.broadcast_object_list(issued, src=0)
            entries = issued[0]
            batches = []
            for index in range(math.ceil(320 / context.dp_size)):
                position = index * context.dp_size + context.dp_rank
                entry = entries[position] if position < len(entries) else entries[0]
                batch = batch_for(panels, [entry], context.device, model_config.tickers)
                batch["weight"] = torch.tensor([float(position < len(entries))], device=context.device)
                batches.append(batch)
            report = update(model, wrapped, optimizer, batches, state["step"])
            del batches
            elapsed = time.monotonic() - started
            worst_update = max(worst_update, elapsed * 1.5)
            state["step"] += 1
            state["scheduled"] += 320
            peak = torch.tensor(torch.cuda.max_memory_reserved(), device=context.device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if context.rank == 0:
                record = {**state, **report, "seconds": elapsed, "samples_per_second": 320 / elapsed,
                          "peak_reserved_gib_all_ranks": peak.item() / 2**30}
                with (root / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                for key, value in report.items():
                    writer.add_scalar("train/" + key, value, state["step"])
                writer.add_scalar("system/samples_per_second", 320 / elapsed, state["step"])
                writer.add_scalar("system/peak_reserved_gib_all_ranks", peak.item() / 2**30, state["step"])
                writer.flush()
                print(json.dumps({"event": "update", **record}), flush=True)
            validation_bound = gate["worst_microbatch_seconds"] * len(registry["monitor_line_indices"]) / context.dp_size + 120
            validate_now = (state["step"] == config["parent_step"] + 1 or state["step"] % 5 == 0) and time.time() + validation_bound < arguments.stop_at
            if validate_now:
                summary = validate(model, panels, panel, registry)
                improved = state["best_metric"] is None or summary["macro_spearman"] > state["best_metric"]
                if improved:
                    state["best_metric"] = summary["macro_spearman"]
                if context.rank == 0:
                    with (root / "validation.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": state["step"], "monitoring_subset": True, **summary}) + "\n")
                    writer.add_scalar("validation/macro_spearman", summary["macro_spearman"], state["step"])
                    for key, value in summary["market_tasks"].items():
                        writer.add_scalar("validation/" + key, value, state["step"])
                saved, duration = save(root, model, optimizer, state, sampler, config, best=improved)
                last_saved = state["step"]
                next_save = time.monotonic() + 1800
                if context.rank == 0:
                    writer.add_scalar("system/checkpoint_seconds", duration, state["step"])
                maintain_distributed(root, context)
            elif time.monotonic() >= next_save:
                saved, duration = save(root, model, optimizer, state, sampler, config)
                last_saved = state["step"]
                next_save = time.monotonic() + 1800
                if context.rank == 0:
                    writer.add_scalar("system/checkpoint_seconds", duration, state["step"])
                maintain_distributed(root, context)
        if state["step"] != last_saved:
            save(root, model, optimizer, state, sampler, config)
            maintain_distributed(root, context)
        if context.rank == 0:
            write_json(root / "TRAINING_DONE.json", {**state, "finished_at": time.time(), "test_evaluated": False})
    if context.rank == 0:
        sampler.close()
        writer.close()
    del wrapped, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    context.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--stop-at", type=float, default=0)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path)
    continuation.add_argument("--migrate-from", type=Path)
    parser.add_argument("--warm-start", type=Path)
    run(parser.parse_args())