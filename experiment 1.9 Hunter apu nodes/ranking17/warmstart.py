import json
from pathlib import Path

import torch
from safetensors import safe_open

from ranker.checkpoints import digest


SOURCE_REPO = "YL95/experiment-1.7-asc-gpu-nodes"
SOURCE_REVISION = "ecf683c5cfd3f49c9fbf2ce5e53f2f637dfce9b9"


def merge_parameter(parts, kind, name):
    if kind is None:
        if any(not torch.equal(parts[0], part) for part in parts[1:]):
            raise ValueError(f"Replicated source parameter differs: {name}")
        return parts[0]
    if kind == "qkv":
        components = 2 if name.startswith("film_") else 3
        chunks = [part.chunk(components, dim=0) for part in parts]
        return torch.cat([torch.cat([chunk[index] for chunk in chunks], dim=0)
                          for index in range(components)], dim=0)
    return torch.cat(parts, dim=kind)


def load_warm_start(model, directory):
    directory = Path(directory)
    provenance = json.loads((directory / "download.json").read_text())
    if (provenance["repository"], provenance["revision"]) != (SOURCE_REPO, SOURCE_REVISION):
        raise ValueError("Unexpected warm-start source revision")
    source = json.loads((directory / "latest" / "config.json").read_text())
    state = json.loads((directory / "latest" / "state.json").read_text())
    if source["tp"] != 8 or model.context.tp_size != 4 or state["step"] != 153:
        raise ValueError("Expected the approved step-153 TP8 to TP4 warm start")
    from dataclasses import asdict
    if source["model"] != asdict(model.config):
        raise ValueError("Warm-start model configuration differs")
    paths = [directory / "latest" / f"weights-{shard:02d}.safetensors"
             for shard in (model.context.tp_rank * 2, model.context.tp_rank * 2 + 1)]
    for path in paths:
        expected = provenance["files"][path.relative_to(directory).as_posix()]
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Warm-start shard verification failed: {path.name}")
    with safe_open(paths[0], framework="pt", device="cpu") as first, safe_open(paths[1], framework="pt", device="cpu") as second:
        parameters = dict(model.named_parameters())
        if set(first.keys()) != set(parameters) or set(second.keys()) != set(parameters):
            raise ValueError("Warm-start parameter names differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = merge_parameter([first.get_tensor(name), second.get_tensor(name)],
                                        getattr(parameter, "tp_kind", None), name)
                if value.shape != parameter.shape:
                    raise ValueError(f"Warm-start parameter shape differs: {name}")
                parameter.copy_(value)
    return {"repository": SOURCE_REPO, "revision": SOURCE_REVISION,
            "parent_step": state["step"], "parent_scheduled": state["scheduled"],
            "source_precision": "bfloat16 matrices; float32 vectors",
            "optimizer_reset": True, "sampler_reset": True, "rng_reset": True,
            "possible_sample_reuse": True, "exact_continuation": False}