import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from safetensors.torch import save_file

from ranker.checkpoints import digest
from ranker.data import write_json


def rng_state(device):
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]), "python": random.getstate()}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)
    np.random.set_state((state["numpy"][0], np.array(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:]))
    random.setstate(state["python"])


def verify(directory, names=None):
    directory = Path(directory)
    if not (directory / "COMPLETE").is_file():
        raise ValueError("Checkpoint has no COMPLETE marker")
    if (directory / "COMPLETE").read_text().strip() != digest(directory / "manifest.json"):
        raise ValueError("Checkpoint manifest digest mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in names if names is not None else manifest:
        path = directory / name
        expected = manifest[name]
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint digest mismatch: {name}")
    return manifest


def save(root, model, optimizer, state, sampler, config, best=False):
    context = model.context
    started = time.monotonic()
    destination = root / "checkpoints" / f"step_{state['step']:08d}_{state['scheduled']:012d}"
    stage = destination.with_name(destination.name + ".staging")
    if context.rank == 0:
        stage.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    files = []
    rng_name = f"rng-{context.rank:04d}.pt"
    torch.save(rng_state(context.device), stage / rng_name)
    files.append(rng_name)
    if context.dp_rank == 0:
        name = f"recovery-{context.tp_rank:02d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stage / name)
        files.append(name)
        if best:
            name = f"weights-{context.tp_rank:02d}.safetensors"
            weights = {key: value.detach().to(device="cpu", dtype=torch.bfloat16 if value.ndim >= 2 else torch.float32).contiguous()
                       for key, value in model.state_dict().items()}
            save_file(weights, str(stage / name))
            files.append(name)
            del weights
    if context.rank == 0:
        sampler.save(stage / "sampler.sqlite")
        write_json(stage / "state.json", state)
        write_json(stage / "config.json", config)
        files.extend(("sampler.sqlite", "state.json", "config.json"))
    local = {name: {"bytes": (stage / name).stat().st_size, "sha256": digest(stage / name)} for name in files}
    gathered = [None] * dist.get_world_size() if context.rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if context.rank == 0:
        manifest = {key: value for chunk in gathered for key, value in chunk.items()}
        write_json(stage / "manifest.json", manifest)
        (stage / "COMPLETE").write_text(digest(stage / "manifest.json") + "\n")
        stage.rename(destination)
        entry = {"directory": str(destination.relative_to(root)), "step": state["step"]}
        write_json(root / "latest.json", entry)
        if best:
            write_json(root / "best.json", entry)
    dist.barrier()
    verify(destination, files)
    return destination, time.monotonic() - started


def restore(directory, model, optimizer, config):
    context = model.context
    names = [f"recovery-{context.tp_rank:02d}.pt", f"rng-{context.rank:04d}.pt", "state.json", "config.json", "sampler.sqlite"]
    verify(directory, names)
    if json.loads((directory / "config.json").read_text()) != config:
        raise ValueError("Exact resume configuration mismatch")
    state = json.loads((directory / "state.json").read_text())
    if state["world"] != dist.get_world_size():
        raise ValueError("Exact resume requires the original parallel layout")
    recovery = torch.load(directory / names[0], map_location="cpu", weights_only=True)
    model.load_state_dict(recovery["model"])
    optimizer.load_state_dict(recovery["optimizer"])
    del recovery
    restore_rng(torch.load(directory / names[1], map_location="cpu", weights_only=True), context.device)
    return state


def backup(directory, destination):
    manifest = verify(directory)
    if destination.exists():
        verify(destination)
        return
    stage = destination.with_name(destination.name + ".staging")
    stage.mkdir(parents=True, exist_ok=False)
    for name in list(manifest) + ["manifest.json"]:
        shutil.copy2(directory / name, stage / name)
    shutil.copy2(directory / "COMPLETE", stage / "COMPLETE")
    verify(stage)
    stage.rename(destination)


def tensor_digest(model, optimizer):
    result = hashlib.sha256()
    for value in model.state_dict().values():
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    for state in optimizer.state.values():
        for key in sorted(state):
            value = state[key]
            result.update(value.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
    return result.hexdigest()