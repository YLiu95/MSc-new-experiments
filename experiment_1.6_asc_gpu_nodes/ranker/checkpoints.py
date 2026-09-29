from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sqlite3

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import distributed as dist

from .data import write_json
from .model import ModelConfig, RankingModel


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(directory: Path) -> dict:
    if not (directory / "COMPLETE").is_file():
        raise ValueError(f"Incomplete checkpoint: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint integrity failed for {name}")
    with sqlite3.connect(f"file:{directory / 'sampler.sqlite'}?mode=ro", uri=True) as ledger:
        if ledger.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Sampler ledger is damaged")
    return manifest


def pointer(root: Path, name: str) -> Path:
    if name not in ("best", "latest"):
        raise ValueError("Only best/latest pointers may be read")
    directory = (root / json.loads((root / f"{name}.json").read_text())["directory"]).resolve()
    directory.relative_to((root / "checkpoints").resolve())
    verify(directory)
    return directory


def save_checkpoint(root: Path, model: RankingModel, optimizer: torch.optim.Optimizer,
                    state: dict, sampler, config: dict, is_best: bool,
                    rank: int = 0, world: int = 1, device: torch.device = torch.device("cpu")) -> Path:
    checkpoint_root = root / "checkpoints"
    name = f"step_{state['step']:08d}_attempt_{state['scheduled']:012d}"
    stage = checkpoint_root / f".{name}.staging"
    directory = checkpoint_root / name
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if stage.exists() or directory.exists():
            raise ValueError("Refusing to replace an existing or interrupted checkpoint")
        stage.mkdir()
    if world > 1:
        dist.barrier()
    numpy_state = np.random.get_state()
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
           "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
           "python": random.getstate()}
    torch.save(rng, stage / f"rng-{rank:04d}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state},
                   stage / "recovery.pt")
        sampler.save(stage / "sampler.sqlite")
        weights = {}
        if is_best:
            for name, value in model.state_dict().items():
                matrix = value.ndim >= 2 and "norm" not in name
                weights[name] = value.detach().to(device="cpu", dtype=torch.bfloat16 if matrix else torch.float32).contiguous()
            save_file(weights, str(stage / "weights.safetensors"))
        write_json(stage / "config.json", config)
        write_json(stage / "state.json", state)
        saved = torch.load(stage / "recovery.pt", map_location="cpu", weights_only=True)
        if saved["state"] != state or set(saved["model"]) != set(model.state_dict()):
            raise ValueError("Recovery checkpoint did not round-trip")
        if is_best:
            exported = load_file(stage / "weights.safetensors", device="cpu")
            if set(exported) != set(saved["model"]):
                raise ValueError("Inference export has a missing parameter")
            with torch.random.fork_rng():
                reference = RankingModel(ModelConfig(**config["model"])).eval()
                inference = RankingModel(ModelConfig(**config["model"])).eval()
                reference.load_state_dict(saved["model"])
                inference.load_state_dict(exported)
                sample = torch.linspace(-1, 1, 3 * 64).reshape(1, 3, 64)
                identities = torch.zeros((1, 3), dtype=torch.long)
                markets = torch.zeros(1, dtype=torch.long)
                task = torch.tensor([[64, 7, 3]])
                with torch.no_grad():
                    original = reference(sample, identities, markets, task)[0]
                    restored = inference(sample, identities, markets, task)[0]
                torch.testing.assert_close(original, restored, atol=0.02, rtol=0.02)
        files = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                 for path in stage.iterdir() if path.is_file()}
        write_json(stage / "manifest.json", {"files": files, "step": state["step"], "is_best": is_best,
                                             "export_tolerance": {"atol": 0.02, "rtol": 0.02} if is_best else None})
        (stage / "COMPLETE").write_text("complete\n")
        verify(stage)
        stage.replace(directory)
        entry = {"directory": directory.relative_to(root).as_posix(), "step": state["step"],
                 "scheduled": state["scheduled"]}
        write_json(root / "latest.json", entry)
        if is_best:
            write_json(root / "best.json", entry)
    if world > 1:
        dist.barrier()
    return directory


def restore_checkpoint(directory: Path, model: RankingModel, optimizer: torch.optim.Optimizer,
                       config: dict, rank: int = 0, world: int = 1,
                       device: torch.device = torch.device("cpu")) -> tuple[dict, Path]:
    verify(directory)
    if json.loads((directory / "config.json").read_text()) != config:
        raise ValueError("Exact resume requires matching dataset, software, model, and training configuration")
    recovery = torch.load(directory / "recovery.pt", map_location="cpu", weights_only=True)
    if recovery["state"]["world"] != world:
        raise ValueError("Exact sampler/optimizer resume requires the original DDP layout")
    model.load_state_dict(recovery["model"])
    optimizer.load_state_dict(recovery["optimizer"])
    rng = torch.load(directory / f"rng-{rank:04d}.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], device)
    np.random.set_state((rng["numpy"][0], np.asarray(rng["numpy"][1], dtype=np.uint32),
                         *rng["numpy"][2:]))
    random.setstate(rng["python"])
    return recovery["state"], directory / "sampler.sqlite"