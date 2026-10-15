import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import uuid

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file

from ascgpu.model import ParallelContext
from ranker.contract import MARKETS, REVISION
from ranker.data import MarketPanel, PREPARATION_POLICY, write_json
from ranker.sampler import Sampler
from ranker.train import optimizer_for
from ranking17.model import ModelConfig, RankingModel
from ranking17.offline import export_checkpoint
from ranking17.recovery import digest, save, verify
from ranking17.train import batch_for, memory_record, update, validate


STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def next_action(now, checkpoint_at, validation_at, end_at, final_saved, validation_seconds=120, unsaved=True):
    if not unsaved:
        return "update"
    if now >= validation_at and now + validation_seconds + 90 < end_at:
        return "validate"
    if now >= checkpoint_at or (not final_saved and now >= end_at - 90):
        return "save"
    return "update"


def retain(root):
    selected = {}
    for label in ("latest", "best"):
        pointer = root / f"{label}.json"
        if pointer.exists():
            checkpoint = (root / json.loads(pointer.read_text())["directory"]).resolve()
            checkpoint.relative_to((root / "checkpoints").resolve())
            verify(checkpoint, workers=4)
            selected[label] = checkpoint
            export_checkpoint(root, checkpoint, label)
    if "latest" not in selected:
        raise ValueError("No verified latest checkpoint")
    protected = set(selected.values())
    for checkpoint in (root / "checkpoints").glob("step_*"):
        if checkpoint.is_dir() and checkpoint.resolve() not in protected and not checkpoint.name.endswith(".staging"):
            shutil.rmtree(checkpoint)
    names = {checkpoint.name for checkpoint in protected}
    for export in (root / "exports").glob("step_*"):
        if export.is_dir() and export.name not in names and not export.name.endswith(".staging"):
            shutil.rmtree(export)
    write_json(root / "reports" / "checkpoint_verification.json", {
        "verified": True, "independent_failure_domain": False,
        "checkpoints": {label: {"directory": path.name, "manifest_sha256": digest(path / "manifest.json")}
                        for label, path in selected.items()}, "time": time.time()})


def checkpoint(root, model, optimizer, state, sampler, config, best=False):
    saved, duration = save(root, model, optimizer, state, sampler, config, best=best)
    status = [None]
    if model.context.rank == 0:
        try:
            retain(root)
            status[0] = {"event": "checkpoint_verified", "step": state["step"], "best": best,
                         "directory": saved.name, "save_seconds": duration}
            print(json.dumps(status[0]), flush=True)
        except Exception as error:
            status[0] = {"error": str(error)}
    dist.broadcast_object_list(status, src=0)
    if "error" in status[0]:
        raise RuntimeError(status[0]["error"])
    return time.time()


def main(arguments):
    root, panel = arguments.root, arguments.panel
    config = json.loads((arguments.initial / "config.json").read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    cpus = set(range(local_rank * 24, (local_rank + 1) * 24))
    if not cpus <= os.sched_getaffinity(0):
        raise RuntimeError("Expected access to all 96 physical CPU cores")
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(4)
    context = ParallelContext.initialize(1)
    if context.dp_size != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("Experiment 1.9 requires one full four-APU node")
    probe = torch.tensor(context.rank + 1, device=context.device)
    dist.all_reduce(probe)
    if probe.item() != 10:
        raise RuntimeError("Four-rank RCCL collective failed")
    torch.use_deterministic_algorithms(True)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    for name, key in (("meta.json", "data_meta_sha256"), ("vocabulary.json", "vocabulary_sha256")):
        if digest(panel / name) != config[key]:
            raise ValueError("Parent data fingerprint changed")
    metadata = json.loads((panel / "meta.json").read_text())
    if metadata["revision"] != REVISION or metadata["preparation_policy"] != PREPARATION_POLICY:
        raise ValueError("Corrected pinned data required")
    model_config = ModelConfig(**config["model"])
    model = RankingModel(model_config, context)
    if model.parameter_count() != 1069302417:
        raise ValueError("Expected verified 1.069B architecture")
    verify(arguments.initial, ("weights-00.safetensors", "config.json", "state.json"))
    model.load_state_dict(load_file(arguments.initial / "weights-00.safetensors"))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], process_group=context.dp_group,
                                      broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = optimizer_for(model, context.device)
    values = torch.linspace(-1, 1, 128 * 512, device=context.device).reshape(1, 128, 512)
    synthetic = {"inputs": values, "ticker_ids": torch.arange(128, device=context.device)[None],
                 "market": torch.zeros(1, dtype=torch.long, device=context.device),
                 "task": torch.tensor([[512, 90, 128]], device=context.device),
                 "labels": torch.arange(128, device=context.device).float()[None],
                 "observed": torch.ones(1, 128, dtype=torch.bool, device=context.device),
                 "weight": torch.ones(1, device=context.device)}
    started = time.monotonic()
    gate_loss = update(model, wrapped, optimizer, [synthetic], 0)
    gate_seconds = time.monotonic() - started
    memory = memory_record(context, "full_model_backward_adam")
    if any(item["peak_reserved_bytes"] > 0.9 * item["hip_total_bytes"] for item in memory):
        raise RuntimeError("Worst-case memory exceeds the 90 percent gate")
    del synthetic, values
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    model.load_state_dict(load_file(arguments.initial / "weights-00.safetensors"))
    torch.cuda.empty_cache()
    seed = model_config.seed + context.dp_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    panels = [MarketPanel(panel, market) for market in MARKETS]
    registry = json.loads((panel / "evaluation" / "val-primary.json").read_text())
    if digest(panel / "evaluation" / registry["file"]) != config["validation_sha256"]:
        raise ValueError("Parent monitoring registry changed")
    sampler = Sampler(panels, root / "ledgers" / f"{uuid.uuid4().hex}.sqlite", model_config.seed) if context.rank == 0 else None
    writer = SummaryWriter(str(root / "runs")) if context.rank == 0 else None
    state = {"step": 0, "scheduled": 0, "best_metric": None, "world": 4, "warm_start_parent_step": 305}
    config["allocation_end_epoch"] = arguments.end_at
    if context.rank == 0:
        write_json(root / "config.json", config)
        write_json(root / "reports" / "preflight.json", {"passed": True, "world": 4, "tp": 1, "dp": 4,
                   "parameters": model.parameter_count(), "worst_microbatch_seconds": gate_seconds,
                   "synthetic_loss": gate_loss["loss"], "memory_phases": memory,
                   "synthetic_update_rolled_back": True, "resume_bitwise_tested": False})
        print(json.dumps({"event": "preflight_passed", "parameters": model.parameter_count(), "tp": 1, "dp": 4}), flush=True)
    summary = validate(model, panels, panel, registry)
    metric = summary["macro_spearman"]
    if metric is None or not math.isfinite(metric):
        raise RuntimeError("Initial monitoring validation is nonfinite")
    state["best_metric"] = metric
    if context.rank == 0:
        with (root / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": 0, "monitoring_subset": True, **summary}) + "\n")
        writer.add_scalar("validation/macro_spearman", metric, 0)
        print(json.dumps({"event": "initial_validation", "step": 0, "macro_spearman": metric}), flush=True)
    last_checkpoint = checkpoint(root, model, optimizer, state, sampler, config, best=True)
    saved_step = state["step"]
    last_validation = time.time()
    validation_seconds = 120.0
    final_saved = False
    while True:
        stopping = torch.tensor(int(STOP), device=context.device)
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        decision = [None]
        if context.rank == 0:
            decision[0] = "stop" if stopping.item() else next_action(
                time.time(), last_checkpoint + 120, last_validation + 300,
                arguments.end_at, final_saved, validation_seconds, state["step"] != saved_step)
        dist.broadcast_object_list(decision, src=0)
        action = decision[0]
        if action == "stop":
            if context.rank == 0:
                writer.flush()
                print(json.dumps({"event": "allocation_termination_requested", **state}), flush=True)
            break
        if action in ("validate", "save"):
            improved = False
            if action == "validate":
                started = time.monotonic()
                summary = validate(model, panels, panel, registry)
                validation_seconds = max(validation_seconds, time.monotonic() - started)
                metric = summary["macro_spearman"]
                improved = metric is not None and math.isfinite(metric) and metric > state["best_metric"]
                if improved:
                    state["best_metric"] = metric
                if context.rank == 0:
                    with (root / "validation.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": state["step"], "monitoring_subset": True, **summary}) + "\n")
                    writer.add_scalar("validation/macro_spearman", metric, state["step"])
                    print(json.dumps({"event": "validation", "step": state["step"], "macro_spearman": metric}), flush=True)
                last_validation = time.time()
            last_checkpoint = checkpoint(root, model, optimizer, state, sampler, config, best=improved)
            saved_step = state["step"]
            final_saved = final_saved or last_checkpoint >= arguments.end_at - 90
            continue
        started = time.monotonic()
        issued = [sampler.issue(320) if context.rank == 0 else None]
        dist.broadcast_object_list(issued, src=0)
        batches = []
        for position in range(context.dp_rank, 320, context.dp_size):
            batch = batch_for(panels, [issued[0][position]], context.device, model_config.tickers)
            batch["weight"] = torch.ones(1, device=context.device)
            batches.append(batch)
        report = update(model, wrapped, optimizer, batches, state["step"])
        del batches
        state["step"] += 1
        state["scheduled"] += 320
        elapsed = time.monotonic() - started
        if context.rank == 0:
            record = {**state, **report, "seconds": elapsed, "samples_per_second": 320 / elapsed}
            with (root / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            for name, value in report.items():
                writer.add_scalar("train/" + name, value, state["step"])
            writer.add_scalar("system/samples_per_second", 320 / elapsed, state["step"])
            writer.flush()
            print(json.dumps({"event": "update", **record}), flush=True)
    if context.rank == 0:
        sampler.close()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--end-at", type=float, required=True)
    main(parser.parse_args())