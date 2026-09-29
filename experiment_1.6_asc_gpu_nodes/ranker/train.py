from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .checkpoints import digest, pointer, restore_checkpoint, save_checkpoint
from .contract import MARKETS, REVISION, Task
from .data import MarketPanel, PREPARATION_POLICY, write_json
from .evaluate import evaluate_model
from .losses import pairwise_logistic
from .model import ModelConfig, RankingModel
from .sampler import Assignment, Sampler


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Training deadlines require a timezone")
    return parsed.timestamp()


def learning_rate(step: int) -> float:
    position = step + 1
    if position <= 1000:
        return 2e-4 * position / 1000
    phase = min(1.0, (position - 1000) / 29000)
    return 4e-6 + (2e-4 - 4e-6) * (1 + math.cos(math.pi * phase)) / 2


def optimizer_for(model: RankingModel, device: torch.device) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=2e-7, betas=(0.9, 0.999), eps=1e-8, fused=device.type == "cuda")


class AssignedBatches(Dataset):
    def __init__(self, panels: list[MarketPanel], shared, unknown_id: int):
        self.panels = panels
        self.shared = shared
        self.unknown_id = unknown_id

    def __len__(self) -> int:
        return len(self.shared)

    def __getitem__(self, index: int) -> dict:
        assignments = self.shared[index]
        if assignments is None:
            raise ValueError("Attempted to load samples before centralized assignment")
        samples = [self.panels[entry.market].sample(entry.cutoff, Task(*entry.task), entry.tickers, self.unknown_id)
                   for entry, _ in assignments]
        return {"inputs": np.stack([sample["inputs"] for sample in samples]),
                "ticker_ids": np.stack([sample["ticker_ids"] for sample in samples]),
                "market": np.array([sample["market"] for sample in samples], dtype=np.int64),
                "task": np.array([sample["task"] for sample in samples], dtype=np.int64),
                "labels": np.stack([sample["labels"] for sample in samples]),
                "observed": np.stack([sample["observed"] for sample in samples]),
                "weight": np.array([weight for _, weight in assignments], dtype=np.float32)}


def informative_count(batch: dict) -> int:
    observed = batch["observed"]
    labels = torch.nan_to_num(batch["labels"])
    comparisons = (observed[:, :, None] & observed[:, None, :]
                   & (labels[:, :, None] != labels[:, None, :]))
    return int((torch.triu(comparisons, diagonal=1).any(dim=(1, 2)) & (batch["weight"] > 0)).sum())


def update(model: RankingModel, wrapped, optimizer, batches: list[dict], device: torch.device,
           rank: int, world: int, conditioned: bool, step: int) -> dict:
    local_count = sum(informative_count(batch) for batch in batches)
    global_count = torch.tensor(local_count, dtype=torch.long, device=device)
    if world > 1:
        dist.all_reduce(global_count)
    if not global_count.item():
        return {"informative": 0, "pairs": 0, "skipped": True}
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = rate
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    all_scores = []
    for index, batch in enumerate(batches):
        synchronization = wrapped.no_sync() if world > 1 and index + 1 < len(batches) else nullcontext()
        with synchronization:
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                scores, _, _ = wrapped(batch["inputs"].to(device, non_blocking=True),
                                        batch["ticker_ids"].to(device, non_blocking=True),
                                        batch["market"].to(device, non_blocking=True),
                                        batch["task"].to(device, non_blocking=True), conditioned=conditioned)
                result = pairwise_logistic(scores, batch["labels"].to(device, non_blocking=True),
                                           batch["observed"].to(device, non_blocking=True),
                                           batch["weight"].to(device, non_blocking=True))
                loss = result.total * (world / global_count.item())
            loss.backward()
            totals += torch.stack((result.total.detach(), result.pairs.detach())).to(torch.float64)
            all_scores.append(scores.detach().float().var(unbiased=False))
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if world > 1:
        dist.all_reduce(totals)
    if not torch.isfinite(totals).all() or not torch.isfinite(gradient_norm):
        raise RuntimeError("Nonfinite ranking loss/gradient; refusing to advance a corrupted update")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"informative": int(global_count.item()), "pairs": int(totals[1].item()),
            "loss": float(totals[0].item()) / global_count.item(), "lr": rate,
            "gradient_norm": float(gradient_norm.item()),
            "score_variance": float(torch.stack(all_scores).mean().item()), "skipped": False}


def source_digest(directory: Path) -> str:
    digest_value = hashlib.sha256()
    for path in sorted((directory / "ranker").glob("*.py")):
        digest_value.update(path.name.encode())
        digest_value.update(path.read_bytes())
    return digest_value.hexdigest()


def train(arguments) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if arguments.global_batch <= 0 or arguments.global_batch % 4 or (device.type == "cuda" and arguments.global_batch != 320):
        raise ValueError("GPU training requires the registered 320 real samples and complete four-basket visits")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(signum, request_stop)
    root = arguments.root / arguments.arm
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = json.loads((arguments.panel / "meta.json").read_text())
    if metadata["revision"] != REVISION or metadata.get("preparation_policy") != PREPARATION_POLICY:
        raise ValueError("Training needs the pinned revision with source-imputed prices excluded")
    registries = {name: json.loads((arguments.panel / "evaluation" / f"val-{name}.json").read_text())
                  for name in ("primary", "heldout")}
    if any(not (arguments.panel / "evaluation" / value["file"]).is_file() for value in registries.values()):
        raise ValueError("Validation manifests must be frozen before training")
    source_dir = Path(__file__).resolve().parents[1]
    config_model = ModelConfig(tickers=metadata["n_tickers"])
    config = {"model": asdict(config_model), "arm": arguments.arm, "seed": arguments.seed,
              "data": {"revision": REVISION, "preparation_policy": PREPARATION_POLICY,
                       "scale": metadata["return_scale_pct"],
                       "vocabulary_sha256": digest(arguments.panel / "vocabulary.json")},
              "validation": {name: value["sha256"] for name, value in registries.items()},
              "software": {"torch": str(torch.__version__), "numpy": np.__version__,
                           "lock_sha256": digest(source_dir / "requirements.lock.txt"),
                           "source_sha256": source_digest(source_dir)},
              "training": {"global_batch": arguments.global_batch, "microbatch": arguments.microbatch,
                           "workers": arguments.workers, "max_updates": 30000, "warmup_updates": 1000,
                           "full_every": 3000, "monitor_every": 500}}
    panels = [MarketPanel(arguments.panel, market) for market in MARKETS]
    torch.manual_seed(arguments.seed)
    model = RankingModel(config_model).to(device)
    optimizer = optimizer_for(model, device)
    torch.manual_seed(arguments.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(arguments.seed + rank)
    conditioned = arguments.arm != "B"
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                      find_unused_parameters=not conditioned, broadcast_buffers=False) if world > 1 else model
    state = {"step": 0, "scheduled": 0, "informative": 0, "pairs": 0, "skipped_batches": 0,
             "consecutive_empty": 0, "best_metric": None, "best_step": None, "world": world,
             "next_monitor": 500, "next_full": 3000, "arm": arguments.arm,
             "stop_at": arguments.stop_at, "backup_by": arguments.backup_by}
    sampler = None
    ledger = root / "ledgers" / f"sampler-{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex}.sqlite"
    if arguments.resume:
        state, saved_ledger = restore_checkpoint(arguments.resume, model, optimizer, config, rank, world, device)
        state["stop_at"] = arguments.stop_at
        state["backup_by"] = arguments.backup_by
        if rank == 0:
            sampler = Sampler.restore(panels, saved_ledger, ledger, arguments.seed,
                                      Task(256, 7, 64) if arguments.arm == "A" else None)
            if sampler.issued != state["scheduled"]:
                raise ValueError("Recovery sampler and completed sample count disagree")
            with (root / "restarts.jsonl").open("a") as stream:
                stream.write(json.dumps({"from": str(arguments.resume), "at": time.time(),
                                         "scheduled_in_checkpoint": state["scheduled"],
                                         "physical_replay_after_checkpoint_possible": True}) + "\n")
    elif rank == 0:
        sampler = Sampler(panels, ledger, arguments.seed, Task(256, 7, 64) if arguments.arm == "A" else None)
    if world > 1:
        dist.barrier()
    writer = SummaryWriter(str(root / "runs")) if rank == 0 else None
    microsteps = math.ceil(arguments.global_batch / (world * arguments.microbatch))
    manager = multiprocessing.Manager() if arguments.workers else None
    shared = manager.list([None] * microsteps) if manager else [None] * microsteps
    loader_options = {"num_workers": arguments.workers, "pin_memory": device.type == "cuda"}
    if arguments.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(AssignedBatches(panels, shared, metadata["n_tickers"]), batch_size=None, **loader_options)
    train_stop = timestamp(arguments.stop_at)
    backup_due = timestamp(arguments.backup_by)
    if train_stop >= backup_due:
        raise ValueError("Training must stop before the backup-completion deadline")
    if rank == 0:
        write_json(root / "config.json", config)
        print(json.dumps({"event": "training_start", "arm": arguments.arm, "rank_count": world,
                          "parameters": sum(parameter.numel() for parameter in model.parameters()),
                          "scheduled": state["scheduled"], "step": state["step"],
                          "stop_at": arguments.stop_at, "backup_by": arguments.backup_by}), flush=True)
    last_saved = state["step"] if arguments.resume else -1
    last_saved_attempts = state["scheduled"] if arguments.resume else -1
    last_full = state["step"] if arguments.resume else -1
    next_checkpoint = time.monotonic() + 1800

    def validation(full: bool) -> bool:
        nonlocal last_saved, last_saved_attempts, last_full, next_checkpoint
        registry = registries["primary"]
        summary = evaluate_model(model, panels, arguments.panel, arguments.panel / "evaluation", registry,
                                 device, rank, world, monitoring=not full)
        improved = False
        if rank == 0:
            score = summary["model"]["macro_spearman"]
            if full and (state["best_metric"] is None or score > state["best_metric"]):
                state["best_metric"], state["best_step"] = score, state["step"]
                improved = True
            writer.add_scalar("validation/primary_macro_spearman" if full else "monitor/primary_macro_spearman",
                              score, state["step"])
            writer.flush()
            with (root / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state["step"], "full": full,
                                         "improved": improved, "results": summary}) + "\n")
            print(json.dumps({"event": "validation", "arm": arguments.arm, "step": state["step"],
                              "full": full, "macro_spearman": score, "improved": improved}), flush=True)
        decision = [state["best_metric"] if rank == 0 else None, state["best_step"] if rank == 0 else None,
                    improved if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(decision, src=0)
        state["best_metric"], state["best_step"], improved = decision
        if full:
            last_full = state["step"]
            save_checkpoint(root, model, optimizer, state, sampler, config, improved, rank, world, device)
            last_saved = state["step"]
            last_saved_attempts = state["scheduled"]
            next_checkpoint = time.monotonic() + 1800
        return improved

    try:
        if state["best_metric"] is None:
            validation(full=True)
        while state["step"] < arguments.max_steps:
            stopping = torch.tensor(int(STOP_REQUESTED or time.time() >= train_stop
                                        or (arguments.stop_file is not None and arguments.stop_file.exists())), device=device)
            if world > 1:
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                break
            assignments = [sampler.issue(arguments.global_batch) if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(assignments, src=0)
            entries = assignments[0]
            groups = [entries[index:index + arguments.microbatch]
                      for index in range(0, len(entries), arguments.microbatch)]
            for index in range(microsteps):
                position = index * world + rank
                shared[index] = ([(entry, 1.0) for entry in groups[position]] if position < len(groups)
                                 else [(entries[0], 0.0)] * arguments.microbatch)
            batches = list(loader)
            started = time.monotonic()
            report = update(model, wrapped, optimizer, batches, device, rank, world,
                            conditioned, state["step"])
            state["scheduled"] += arguments.global_batch
            state["informative"] += report["informative"]
            state["pairs"] += report["pairs"]
            if report["skipped"]:
                state["skipped_batches"] += 1
                state["consecutive_empty"] += 1
            else:
                state["step"] += 1
                state["consecutive_empty"] = 0
            if rank == 0 and (state["step"] % 20 == 0 or report["skipped"]):
                record = {"step": state["step"], "scheduled": state["scheduled"],
                          "seconds": time.monotonic() - started, **report}
                with (root / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                for name in ("loss", "lr", "gradient_norm", "score_variance", "informative", "pairs"):
                    if name in report:
                        writer.add_scalar(f"train/{name}", report[name], state["step"])
                if device.type == "cuda":
                    writer.add_scalar("system/peak_gpu_reserved_gib", torch.cuda.max_memory_reserved(device) / 2**30,
                                      state["step"])
                writer.flush()
                print(json.dumps({"event": "optimizer_step" if not report["skipped"] else "empty_batch",
                                  **record}), flush=True)
            if state["consecutive_empty"] >= 100:
                save_checkpoint(root, model, optimizer, state, sampler, config, False, rank, world, device)
                raise RuntimeError("100 consecutive global batches had no informative labels")
            if not report["skipped"] and state["step"] >= state["next_monitor"]:
                state["next_monitor"] += 500
                validation(full=False)
            if not report["skipped"] and state["step"] >= state["next_full"]:
                state["next_full"] += 3000
                validation(full=True)
            elif time.monotonic() >= next_checkpoint and state["scheduled"] != last_saved_attempts:
                save_checkpoint(root, model, optimizer, state, sampler, config, False, rank, world, device)
                last_saved = state["step"]
                last_saved_attempts = state["scheduled"]
                next_checkpoint = time.monotonic() + 1800
        if state["step"] != last_full:
            validation(full=True)
        elif state["scheduled"] != last_saved_attempts:
            save_checkpoint(root, model, optimizer, state, sampler, config, False, rank, world, device)
        if rank == 0:
            writer.close()
            write_json(root / "reports" / "training_summary.json", {**state,
                       "full_validation_selection": True, "test_evaluated": False,
                       "completed_at_unix": time.time(), "best": json.loads((root / "best.json").read_text()),
                       "latest": json.loads((root / "latest.json").read_text())})
            write_json(root / "TRAINING_DONE.json", {"step": state["step"], "scheduled": state["scheduled"],
                                                      "completed_at_unix": time.time()})
    finally:
        if rank == 0 and sampler is not None:
            sampler.close()
        if manager:
            manager.shutdown()
        if world > 1:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--backup-by", required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--global-batch", type=int, default=320)
    parser.add_argument("--microbatch", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-file", type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()