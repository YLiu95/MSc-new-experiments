import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ascgpu.model import ParallelContext
from ranker.data import write_json
from ranking17.model import ModelConfig, RankingModel
from ranking17.recovery import digest, verify
from ranking17.warmstart import merge_parameter


TEMPORAL = (0, 19)
CROSS = (11,)
PARENT_REVISION = "586d6a4eba7f233b1070fc59acfef8857aad6444"


def parameter_count(config):
    width, feedforward = config.width, config.feedforward
    block = 4 * width**2 + 2 * width * feedforward + 9 * width + feedforward
    return (config.temporal_depth + config.cross_depth) * block + (config.tickers + 477) * width + 10897


def source_name(name, temporal=TEMPORAL, cross=CROSS):
    components = name.split(".")
    if components[0] in ("temporal", "cross"):
        selected = temporal if components[0] == "temporal" else cross
        components[1] = str(selected[int(components[1])])
    return ".".join(components)


def load_slice(model, parent, temporal=TEMPORAL, cross=CROSS):
    config = json.loads((parent / "config.json").read_text())
    if config["tp"] != 4 or model.context.tp_size != 1:
        raise ValueError("Expected TP4 source and unsharded target")
    expected = dict(config["model"], temporal_depth=len(temporal), cross_depth=len(cross))
    if asdict(model.config) != expected:
        raise ValueError("Only depth pruning is permitted")
    for selected, depth in ((temporal, config["model"]["temporal_depth"]), (cross, config["model"]["cross_depth"])):
        if not selected or tuple(sorted(set(selected))) != tuple(selected) or min(selected) < 0 or max(selected) >= depth:
            raise ValueError("Layer selection must be increasing, unique and in range")
    mapping = {}
    with ExitStack() as stack, torch.no_grad():
        shards = [stack.enter_context(safe_open(parent / f"weights-{shard:02d}.safetensors", framework="pt", device="cpu"))
                  for shard in range(config["tp"])]
        for name, parameter in model.named_parameters():
            original = source_name(name, temporal, cross)
            parts = [shard.get_tensor(original) for shard in shards]
            value = merge_parameter(parts, getattr(parameter, "tp_kind", None), original)
            if value.shape != parameter.shape:
                raise ValueError(f"Sliced tensor shape mismatch: {name}")
            parameter.copy_(value)
            mapping[name] = original
    return mapping


def extract(parent, output):
    if output.exists():
        verify(output, workers=4)
        return json.loads((output / "config.json").read_text())
    verify(parent, workers=4)
    source = json.loads((parent / "config.json").read_text())
    state = json.loads((parent / "state.json").read_text())
    if source["experiment"] != "1.7.1" or state["step"] != 305:
        raise ValueError("Expected Experiment 1.7.1 best step 305")
    model_config = replace(ModelConfig(**source["model"]), temporal_depth=2, cross_depth=1)
    if parameter_count(model_config) != 1069302417:
        raise ValueError("Unexpected pruned architecture")
    torch.set_num_threads(8)
    model = RankingModel(model_config, ParallelContext())
    mapping = load_slice(model, parent)
    if model.parameter_count() != parameter_count(model_config):
        raise ValueError("Analytic and instantiated parameter counts differ")
    config = {key: source[key] for key in ("data_meta_sha256", "vocabulary_sha256", "validation_sha256")}
    config.update(experiment="1.9", model=asdict(model_config), parameters=model.parameter_count(), tp=1, dp=4,
                  global_batch=320, microbatch=1, parent_step=305, optimizer_reset=True, rng_reset=True,
                  sampler_reset=True, possible_sample_reuse=True, exact_continuation=False,
                  source_repository="YL95/experiment-1.7.1-hunter-apu-nodes", source_revision=PARENT_REVISION,
                  parent_manifest_sha256=digest(parent / "manifest.json"),
                  selected_temporal_layers=list(TEMPORAL), selected_cross_layers=list(CROSS),
                  selection_policy="first and last temporal blocks; final cross block; no channel pruning",
                  matrix_compute="bfloat16", master_weights="float32", cutoff_policy="PBS allocation termination")
    stage = output.with_name(output.name + ".staging")
    stage.mkdir(parents=True, exist_ok=False)
    weights = {name: value.detach().to(dtype=torch.bfloat16 if value.ndim >= 2 else torch.float32).contiguous()
               for name, value in model.state_dict().items()}
    save_file(weights, stage / "weights-00.safetensors")
    write_json(stage / "config.json", config)
    write_json(stage / "layer_mapping.json", mapping)
    write_json(stage / "state.json", {"step": 0, "scheduled": 0, "best_metric": None, "parent_step": 305})
    manifest = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in stage.iterdir()}
    write_json(stage / "manifest.json", manifest)
    (stage / "COMPLETE").write_text(digest(stage / "manifest.json") + "\n")
    verify(stage, workers=4)
    stage.rename(output)
    print(json.dumps({"event": "slice_verified", "parameters": model.parameter_count(), "temporal": TEMPORAL,
                      "cross": CROSS, "parent_step": 305}), flush=True)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    extract(arguments.parent, arguments.output)