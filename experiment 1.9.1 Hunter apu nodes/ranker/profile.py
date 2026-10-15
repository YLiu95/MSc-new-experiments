from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoints import digest, save_checkpoint
from .data import write_json
from .losses import pairwise_logistic
from .model import ModelConfig, RankingModel
from .train import optimizer_for


class SyntheticLedger:
    def save(self, path: Path) -> None:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE profile_only (id INTEGER PRIMARY KEY, synthetic INTEGER)")
            connection.execute("INSERT INTO profile_only VALUES (1, 1)")


def step(model, wrapped, optimizer, batch: dict, accumulations: int,
         device: torch.device, world: int) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    score_variance = 0.0
    for index in range(accumulations):
        synchronization = wrapped.no_sync() if world > 1 and index + 1 < accumulations else nullcontext()
        with synchronization:
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                scores = wrapped(batch["inputs"], batch["ids"], batch["markets"], batch["tasks"])[0]
                losses = pairwise_logistic(scores, batch["labels"], batch["observed"])
                loss = losses.total / (len(batch["inputs"]) * accumulations)
            loss.backward()
            score_variance += float(scores.detach().float().var(unbiased=False).item()) / accumulations
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    if not math.isfinite(float(loss.item())) or not math.isfinite(gradient_norm) or score_variance <= 0:
        raise RuntimeError("The synthetic profile produced nonfinite gradients or collapsed ranking scores")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return gradient_norm, score_variance


def synthetic(config: ModelConfig, device: torch.device, microbatch: int, length: int, horizon: int,
              basket: int) -> dict:
    generator = torch.Generator().manual_seed(1337)
    inputs = torch.randn((microbatch, basket, length), generator=generator).to(device)
    identities = torch.arange(basket, device=device).repeat(microbatch, 1) % config.tickers
    labels = torch.arange(basket, dtype=torch.float32, device=device)[None].repeat(microbatch, 1)
    for index in range(microbatch):
        labels[index] = labels[index].roll(index * 7)
    return {"inputs": inputs, "ids": identities,
            "markets": torch.zeros(microbatch, dtype=torch.long, device=device),
            "tasks": torch.tensor([[length, horizon, basket]] * microbatch, dtype=torch.long, device=device),
            "labels": labels, "observed": torch.ones((microbatch, basket), dtype=torch.bool, device=device)}


def profile(root: Path, warmup: int, measured: int, accumulations: int) -> dict | None:
    if not torch.cuda.is_available():
        raise ValueError("The worst-case profile must run inside a GPU allocation")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    metadata = json.loads((root / "panel" / "meta.json").read_text())
    config = ModelConfig(tickers=metadata["n_tickers"])
    torch.manual_seed(1337)
    model = RankingModel(config).to(device).train()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    optimizer = optimizer_for(model, device)
    small = synthetic(config, device, 4, 64, 1, 3)
    worst = synthetic(config, device, 4, 512, 90, 128)
    step(model, wrapped, optimizer, small, 1, device, world)
    for _ in range(warmup):
        step(model, wrapped, optimizer, worst, accumulations, device, world)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    gradient_norms, variances = [], []
    for _ in range(measured):
        norm, variance = step(model, wrapped, optimizer, worst, accumulations, device, world)
        gradient_norms.append(norm)
        variances.append(variance)
    elapsed = time.monotonic() - started
    local = {"rank": rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
             "mean_gradient_norm": sum(gradient_norms) / measured,
             "mean_score_variance": sum(variances) / measured,
             "measured_update_seconds": elapsed / measured}
    gathered = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    destination = root / "profiling" / f"checkpoint-{os.environ.get('SLURM_JOB_ID', 'local')}"
    checkpoint_state = {"step": warmup + measured, "scheduled": (warmup + measured) * 320,
                        "world": world, "synthetic_profile_not_training": True}
    checkpoint_config = {"model": asdict(config), "purpose": "synthetic_worst_case_only",
                         "software_lock_sha256": digest(Path(__file__).resolve().parents[1] / "requirements.lock.txt")}
    checkpoint_start = time.monotonic()
    save_checkpoint(destination, model, optimizer, checkpoint_state,
                    SyntheticLedger() if rank == 0 else None, checkpoint_config, True, rank, world, device)
    checkpoint_elapsed = time.monotonic() - checkpoint_start
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return None
    result = {"synthetic_not_trained_model": True, "world": world, "warmup": warmup, "measured": measured,
              "accumulations": accumulations, "microbatch": 4, "worst_task": [512, 90, 128],
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "measure_seconds": elapsed, "checkpoint_seconds": checkpoint_elapsed,
              "devices": gathered, "peak_target_gib": 34,
              "under_memory_target": all(entry["peak_reserved_gib"] <= 34 for entry in gathered)}
    write_json(root / "reports" / "gpu_profile.json", result)
    if not result["under_memory_target"]:
        raise RuntimeError("Worst-case profile exceeded 34 GiB; reduce microbatch before production")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--accumulations", type=int, default=10)
    arguments = parser.parse_args()
    profile(arguments.root, arguments.warmup, arguments.measured, arguments.accumulations)


if __name__ == "__main__":
    main()