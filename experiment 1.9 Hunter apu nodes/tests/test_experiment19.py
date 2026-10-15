from dataclasses import asdict, replace
import json

import torch
from safetensors.torch import save_file

from ascgpu.model import ParallelContext
from ranking17.model import ModelConfig, RankingModel
from slice_model import load_slice, parameter_count, source_name


def test_parameter_counts():
    parent = ModelConfig()
    assert parameter_count(parent) == 10193843857
    assert parameter_count(replace(parent, temporal_depth=2, cross_depth=1)) == 1069302417


def test_selected_layers_merge_exactly_and_backpropagate(tmp_path):
    parent = ModelConfig(tickers=8, width=16, heads=4, feedforward=32, temporal_depth=4, cross_depth=3,
                         dropout=0.0, identity_dropout=0.0, rematerialize=False)
    originals = []
    for shard in range(4):
        model = RankingModel(parent, ParallelContext(tp_size=4, tp_rank=shard))
        originals.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        save_file(originals[-1], tmp_path / f"weights-{shard:02d}.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"tp": 4, "model": asdict(parent)}))
    child = RankingModel(replace(parent, temporal_depth=2, cross_depth=1), ParallelContext())
    mapping = load_slice(child, tmp_path, temporal=(0, 3), cross=(2,))
    assert child.parameter_count() == parameter_count(child.config)
    assert mapping["temporal.1.qkv_weight"] == "temporal.3.qkv_weight"
    assert mapping["cross.0.expand.weight"] == "cross.2.expand.weight"
    for name, parameter in child.named_parameters():
        parts = [original[mapping[name]] for original in originals]
        kind = getattr(parameter, "tp_kind", None)
        if kind == "qkv":
            components = 2 if name.startswith("film_") else 3
            expected = torch.cat([torch.cat([part.chunk(components, 0)[component] for part in parts], 0)
                                  for component in range(components)], 0)
        elif kind is None:
            expected = parts[0]
        else:
            expected = torch.cat(parts, kind)
        assert torch.equal(parameter, expected), name
    scores = child(torch.randn(1, 3, 64), torch.tensor([[0, 1, 2]]), torch.tensor([0]), torch.tensor([[64, 1, 3]]))[0]
    scores.square().mean().backward()
    assert scores.shape == (1, 3)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in child.parameters())


def test_layer_mapping():
    assert source_name("temporal.1.norm_attention.weight") == "temporal.19.norm_attention.weight"
    assert source_name("cross.0.qkv_bias") == "cross.11.qkv_bias"
    assert source_name("head_output.weight") == "head_output.weight"


def test_checkpoint_schedule_keeps_training_until_pbs_stops():
    from train19 import next_action
    assert next_action(100, 200, 300, 1000, False) == "update"
    assert next_action(200, 200, 300, 1000, False) == "save"
    assert next_action(300, 400, 300, 1000, False) == "validate"
    assert next_action(910, 999, 300, 1000, False) == "save"
    assert next_action(950, 1030, 300, 1000, True) == "update"
    assert next_action(1001, 1030, 300, 1000, True) == "update"
    assert next_action(300, 200, 300, 1000, False, unsaved=False) == "update"


def test_retention_protects_best_latest_and_inflight(tmp_path):
    from ranker.data import write_json
    from ranking17.recovery import digest
    from train19 import retain
    for step in (0, 1, 2):
        directory = tmp_path / "checkpoints" / f"step_{step:08d}"
        directory.mkdir(parents=True)
        save_file({"weight": torch.ones(2)}, directory / "weights-00.safetensors")
        write_json(directory / "config.json", {"tp": 1})
        write_json(directory / "state.json", {"step": step})
        manifest = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in directory.iterdir()}
        write_json(directory / "manifest.json", manifest)
        (directory / "COMPLETE").write_text(digest(directory / "manifest.json"))
    inflight = tmp_path / "checkpoints" / "step_00000003.staging"
    inflight.mkdir()
    write_json(tmp_path / "best.json", {"directory": "checkpoints/step_00000000"})
    write_json(tmp_path / "latest.json", {"directory": "checkpoints/step_00000002"})
    retain(tmp_path)
    assert (tmp_path / "checkpoints" / "step_00000000").exists()
    assert not (tmp_path / "checkpoints" / "step_00000001").exists()
    assert (tmp_path / "checkpoints" / "step_00000002").exists()
    assert inflight.exists()
    assert json.loads((tmp_path / "exports" / "best.json").read_text())["directory"] == "step_00000000"


def test_publication_archive_rejects_private_or_unsafe_members(tmp_path):
    import io
    import tarfile
    import pytest
    from publish19 import runtime_files
    for name in ("../secret.py", "ranking17/link.py", "recovery-00.pt", ".env"):
        archive = tmp_path / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo(name)
            member.size = 1
            if name.endswith("link.py"):
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            stream.addfile(member, io.BytesIO(b"x"))
        with pytest.raises(ValueError):
            runtime_files(archive)


def test_model_card_has_valid_hub_metadata():
    from huggingface_hub import ModelCard
    from publish19 import SOURCE
    card = ModelCard.load(SOURCE / "MODEL_CARD.md")
    assert "tensorboard" in card.data.tags
    assert "0.08075104483874465" in card.text