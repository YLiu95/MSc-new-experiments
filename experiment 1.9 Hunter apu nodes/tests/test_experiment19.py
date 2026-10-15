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


def test_checkpoint_schedule_saves_with_validation():
    from train19 import next_action
    for now in (100, 120, 200, 240, 299):
        assert next_action(now, 300, 1000, False) == "update"
    assert next_action(300, 300, 1000, False) == "validate"
    assert next_action(600, 600, 1000, False) == "validate"
    assert next_action(300, 300, 1000, False, unsaved=False) == "update"


def test_checkpoint_schedule_preserves_expiry_save_without_early_stop():
    from train19 import next_action
    assert next_action(800, 300, 1000, False) == "update"
    assert next_action(910, 300, 1000, False) == "save"
    assert next_action(910, 300, 1000, False, unsaved=False) == "update"
    assert next_action(950, 300, 1000, True) == "update"
    assert next_action(1001, 300, 1000, True) == "update"


def test_resume_requires_matching_completed_four_worker_checkpoint():
    import pytest
    from train19 import migration_state
    previous = {"tp": 1, "dp": 4, "global_batch": 320}
    state = {"world": 4, "step": 66, "scheduled": 21120, "best_metric": 0.08}
    migrated = migration_state(previous, state, 8)
    assert migrated["world"] == 8 and migrated["step"] == 66
    assert migrated["scheduled"] == 21120 and migrated["best_metric"] == 0.08
    assert migrated["migration_bitwise_continuation"] is False
    with pytest.raises(ValueError):
        migration_state(previous, {**state, "scheduled": 20800}, 8)
    with pytest.raises(ValueError):
        migration_state(previous, state, 4)


def test_resume_snapshot_keeps_prior_checkpoint_immutable(tmp_path):
    import pytest
    from ranker.data import write_json
    from ranking17.recovery import digest
    from train19 import prepare_resume
    previous_root = tmp_path / "prior"
    checkpoint = previous_root / "checkpoints" / "step_00000066_000000021120"
    checkpoint.mkdir(parents=True)
    initial = {"tp": 1, "dp": 4, "global_batch": 320, "model": {"seed": 1337}}
    write_json(checkpoint / "config.json", {**initial, "allocation_end_epoch": 1000})
    write_json(checkpoint / "state.json", {"world": 4, "step": 66, "scheduled": 21120, "best_metric": 0.08})
    (checkpoint / "sampler.sqlite").write_bytes(b"sampler")
    (checkpoint / "recovery-00.pt").write_bytes(b"model and optimizer")
    manifest = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in checkpoint.iterdir()}
    write_json(checkpoint / "manifest.json", manifest)
    (checkpoint / "COMPLETE").write_text(digest(checkpoint / "manifest.json"))
    entry = {"directory": "checkpoints/step_00000066_000000021120", "step": 66}
    for label in ("best", "latest"):
        write_json(previous_root / f"{label}.json", entry)
    root = tmp_path / "new"
    config, state = prepare_resume(root, checkpoint, initial, 2000)
    assert config["dp"] == 8 and config["migration"]["optimizer_preserved"]
    assert state["step"] == 66 and state["world"] == 8
    assert (root / entry["directory"] / "recovery-00.pt").read_bytes() == b"model and optimizer"
    assert (root / entry["directory"] / "recovery-00.pt").stat().st_ino != (checkpoint / "recovery-00.pt").stat().st_ino
    assert (checkpoint / "config.json").read_text() != (root / "config.json").read_text()
    assert json.loads((previous_root / "latest.json").read_text()) == entry
    with pytest.raises(ValueError):
        prepare_resume(tmp_path / "invalid", checkpoint, {**initial, "global_batch": 64}, 2000)


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


def test_restart_requires_current_complete_same_topology_checkpoint(tmp_path):
    import pytest
    from ranker.data import write_json
    from ranking17.recovery import digest
    from train19 import prepare_restart
    initial = {"tp": 1, "dp": 4, "global_batch": 320, "model": {"seed": 1337},
               "optimizer_reset": True, "sampler_reset": True, "possible_sample_reuse": True}
    config = {**initial, "dp": 8, "allocation_end_epoch": 1000, "optimizer_reset": False,
              "sampler_reset": False, "possible_sample_reuse": False, "migration": {"parent": 66}}
    write_json(tmp_path / "config.json", config)
    for step, label in ((328, "best"), (1131, "latest")):
        directory = tmp_path / "checkpoints" / f"step_{step:08d}"
        directory.mkdir(parents=True)
        write_json(directory / "config.json", config)
        write_json(directory / "state.json", {"step": step, "scheduled": step * 320, "world": 8})
        for name in ("recovery-00.pt", "sampler.sqlite", *(f"rng-{rank:04d}.pt" for rank in range(8))):
            (directory / name).write_bytes(b"snapshot")
        manifest = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in directory.iterdir()}
        write_json(directory / "manifest.json", manifest)
        (directory / "COMPLETE").write_text(digest(directory / "manifest.json"))
        write_json(tmp_path / f"{label}.json", {"directory": directory.relative_to(tmp_path).as_posix(), "step": step})
    best_before = (tmp_path / "best.json").read_bytes()
    restored_config, state = prepare_restart(tmp_path, directory, initial, 1000, 8)
    assert restored_config == config and state["step"] == 1131
    assert (tmp_path / "best.json").read_bytes() == best_before
    for changed_initial, deadline, world in (({**initial, "global_batch": 64}, 1000, 8),
                                             (initial, 1001, 8), (initial, 1000, 4)):
        with pytest.raises(ValueError):
            prepare_restart(tmp_path, directory, changed_initial, deadline, world)
    with pytest.raises(ValueError):
        prepare_restart(tmp_path, tmp_path / "checkpoints" / "step_00000328", initial, 1000, 8)
    (directory / "rng-0007.pt").write_bytes(b"corrupt!")
    with pytest.raises(ValueError):
        prepare_restart(tmp_path, directory, initial, 1000, 8)


def test_restart_loader_restores_model_adam_and_per_rank_rng(tmp_path, monkeypatch):
    from copy import deepcopy
    import random
    from types import SimpleNamespace
    import numpy as np
    from ranker.data import write_json
    from ranking17 import recovery
    from train19 import restore
    model = torch.nn.Linear(2, 1)
    model.context = SimpleNamespace(tp_rank=0, rank=3, device=torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected_model, expected_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    config, state = {"tp": 1, "dp": 8}, {"world": 8, "step": 1131, "scheduled": 361920}
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path / "recovery-00.pt")
    numpy_state = np.random.get_state()
    cuda_state = torch.Generator().manual_seed(42).get_state()
    torch.save({"torch": torch.get_rng_state(), "cuda": cuda_state,
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "python": random.getstate()}, tmp_path / "rng-0003.pt")
    expected_torch, expected_numpy, expected_random = torch.rand(3), np.random.rand(3), random.random()
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "state.json", state)
    (tmp_path / "sampler.sqlite").write_bytes(b"sampler")
    manifest = {path.name: {"bytes": path.stat().st_size, "sha256": recovery.digest(path)} for path in tmp_path.iterdir()}
    write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "COMPLETE").write_text(recovery.digest(tmp_path / "manifest.json"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer.state.clear()
    restored_cuda = []
    monkeypatch.setattr(recovery.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: restored_cuda.append(value))
    assert restore(tmp_path, model, optimizer, config) == state
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected_model.items())
    for index, saved in expected_optimizer["state"].items():
        assert all(torch.equal(optimizer.state_dict()["state"][index][key], value) for key, value in saved.items())
    assert torch.equal(torch.rand(3), expected_torch)
    assert np.array_equal(np.random.rand(3), expected_numpy)
    assert random.random() == expected_random
    assert torch.equal(restored_cuda[0], cuda_state)


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


def test_cloud_cleanup_requires_matching_cloud_and_local_bytes(tmp_path):
    import hashlib
    from types import SimpleNamespace
    import pytest
    from cloud_cleanup import remove_verified, verify_candidate
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"weights")
    checksum = hashlib.sha256(b"weights").hexdigest()
    metadata = {"bytes": 7, "digest": checksum}
    remote = SimpleNamespace(size=7, lfs=SimpleNamespace(sha256=checksum))
    entry = verify_candidate(tmp_path, path.name, metadata, remote)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        remove_verified(tmp_path, entry)
    assert path.exists()
    with pytest.raises(ValueError):
        verify_candidate(tmp_path, path.name, metadata, remote)
    path.write_bytes(b"weights")
    remote.lfs.sha256 = "0" * 64
    with pytest.raises(ValueError):
        verify_candidate(tmp_path, path.name, metadata, remote)
    remove_verified(tmp_path, entry)
    assert not path.exists()


def test_cloud_cleanup_rejects_path_escape(tmp_path):
    import pytest
    from cloud_cleanup import safe_path
    for relative in ("../outside", "/outside", "C:/outside", "folder\\outside"):
        with pytest.raises(ValueError):
            safe_path(tmp_path, relative)


def test_cloud_cleanup_supports_legacy_count_receipt():
    from types import SimpleNamespace
    import pytest
    from cloud_cleanup import receipt_files
    remote = {"best/weights.safetensors": SimpleNamespace(size=7, lfs=SimpleNamespace(sha256="a" * 64)),
              "unrelated/weights.safetensors": SimpleNamespace(size=7, lfs=SimpleNamespace(sha256="b" * 64))}
    assert receipt_files({"files": 20, "verified": True}, remote) == {
        "best/weights.safetensors": {"bytes": 7, "sha256": "a" * 64}}
    with pytest.raises(ValueError):
        receipt_files({"files": 20, "verified": False}, remote)


def test_publication_selects_migrated_job_and_freezes_retained_files(tmp_path):
    import pytest
    from publish19 import freeze_snapshot, public_snapshot
    from ranking17.recovery import digest
    job = "job-286615.hunter-pbs01"
    root = tmp_path / "artifacts" / job
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "migration.json").write_text('{"optimizer_preserved": true}')
    (root / "history.jsonl").write_text('{"step": 90}\n')
    directory = root / "exports" / "step_00000090"
    directory.mkdir(parents=True)
    for name, data in {"config.json": b'{"tp": 1}', "state.json": b'{"step": 90}',
                       "weights-00.safetensors": b"weights"}.items():
        (directory / name).write_bytes(data)
    manifest = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in directory.iterdir()}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE").write_text(digest(directory / "manifest.json"))
    for label in ("best", "latest"):
        (root / "exports" / f"{label}.json").write_text(json.dumps({"directory": directory.name}))
    for folder, content in (("source-resume-286615", "original"), ("source-restart-286615", "restarted")):
        source = tmp_path / folder
        source.mkdir()
        for name in ("train19.py", "run.sh"):
            (source / name).write_text(content)
    (root / "reports" / "restart-90.json").write_text(json.dumps({
        "event": "restart_restored", "source_sha256": digest(tmp_path / "source-restart-286615" / "train19.py")}))
    snapshot = public_snapshot(tmp_path, job)
    assert snapshot["job"] == job
    assert snapshot["files"][f"source/{job}/train19.py"]["path"] == "source-restart-286615/train19.py"
    assert snapshot["files"][f"source/{job}/before-restart/train19.py"]["path"] == "source-resume-286615/train19.py"
    assert f"reports/{job}/restart-90.json" in snapshot["files"]
    assert not any("recovery-" in name or "rng-" in name or "sampler.sqlite" in name for name in snapshot["files"])
    frozen = freeze_snapshot(tmp_path, snapshot)
    (directory / "weights-00.safetensors").unlink()
    (root / "history.jsonl").write_text('{"step": 91}\n')
    assert (tmp_path / frozen["files"]["latest/weights-00.safetensors"]["path"]).read_bytes() == b"weights"
    assert (tmp_path / frozen["files"][f"reports/{job}/history.jsonl"]["path"]).read_text() == '{"step": 90}\n'
    with pytest.raises(ValueError):
        public_snapshot(tmp_path, "../outside")


def test_private_recovery_snapshot_and_publication_privacy_guards(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    import publish19
    from ranker.data import write_json
    from ranking17.recovery import digest
    job = "job-286615.hunter-pbs01"
    root = tmp_path / "artifacts" / job
    checkpoint = root / "checkpoints" / "step_00002172"
    checkpoint.mkdir(parents=True)
    for name in ("recovery-00.pt", "weights-00.safetensors", "sampler.sqlite", *(f"rng-{rank:04d}.pt" for rank in range(8))):
        (checkpoint / name).write_bytes(b"private recovery")
    write_json(checkpoint / "state.json", {"step": 2172, "world": 8})
    write_json(checkpoint / "config.json", {"tp": 1, "dp": 8})
    write_json(checkpoint / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                                             for path in checkpoint.iterdir()})
    (checkpoint / "COMPLETE").write_text(digest(checkpoint / "manifest.json"))
    for label in ("best", "latest"):
        write_json(root / f"{label}.json", {"directory": f"checkpoints/{checkpoint.name}", "step": 2172})
    write_json(root / "config.json", {"tp": 1, "dp": 8})
    write_json(root / "reports" / "scheduler.json", {"job_state": "F"})
    snapshot = publish19.recovery_snapshot(tmp_path, job)
    assert snapshot["private_recovery"] is True
    assert f"{job}/checkpoints/{checkpoint.name}/recovery-00.pt" in snapshot["files"]
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: snapshot)
    monkeypatch.setattr(publish19.requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="privacy"):
        publish19.snapshot_at("http://localhost", job)
    assert publish19.snapshot_at("http://localhost", job, private_recovery=True) == snapshot
    api = SimpleNamespace(whoami=lambda: {"name": "YL95"}, create_repo=lambda *args, **kwargs: None,
                          model_info=lambda *args, **kwargs: SimpleNamespace(private=False))
    monkeypatch.setattr(publish19, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(publish19.legacy, "credential", lambda name: "test")
    with pytest.raises(ValueError, match="must be private"):
        publish19.publish_recovery(tmp_path / "cache", "http://localhost", job)
    write_json(root / "reports" / "scheduler.json", {"job_state": "R"})
    with pytest.raises(ValueError, match="finished allocation"):
        publish19.recovery_snapshot(tmp_path, job)
    write_json(root / "reports" / "scheduler.json", {"job_state": "F"})
    (checkpoint / "rng-0007.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        publish19.recovery_snapshot(tmp_path, job)


def test_cloud_publication_verifies_privacy_size_and_digest(tmp_path):
    from types import SimpleNamespace
    import pytest
    from publish19 import verify_cloud_files
    from ranking17.recovery import digest
    path = tmp_path / "recovery.pt"
    path.write_bytes(b"recovery")
    entry = SimpleNamespace(rfilename=path.name, size=path.stat().st_size,
                            lfs=SimpleNamespace(sha256=digest(path)))
    info = SimpleNamespace(private=True, siblings=[entry])
    api = SimpleNamespace(model_info=lambda *args, **kwargs: info)
    assert verify_cloud_files(api, "private/repo", "commit", {path.name: path}, True)[path.name]["bytes"] == 8
    with pytest.raises(ValueError, match="privacy"):
        verify_cloud_files(api, "private/repo", "commit", {path.name: path}, False)
    entry.lfs.sha256 = "bad"
    with pytest.raises(ValueError, match="verification"):
        verify_cloud_files(api, "private/repo", "commit", {path.name: path}, True)


def test_publication_download_verifies_before_replacing_cache(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import hashlib
    from types import SimpleNamespace
    import pytest
    import publish19
    target = tmp_path / "weights.safetensors"
    target.write_bytes(b"previous")
    entry = {"path": "frozen/weights.safetensors", "bytes": 7,
             "sha256": hashlib.sha256(b"weights").hexdigest()}
    response = SimpleNamespace(raise_for_status=lambda: None, iter_content=lambda size: iter([b"corrupt"]))
    monkeypatch.setattr(publish19.requests, "get", lambda *args, **kwargs: nullcontext(response))
    with pytest.raises(ValueError):
        publish19.download_with_progress(tmp_path, target.name, entry, "http://localhost")
    assert target.read_bytes() == b"previous"
    response.iter_content = lambda size: iter([b"weights"])
    assert publish19.download_with_progress(tmp_path, target.name, entry, "http://localhost") == target
    assert target.read_bytes() == b"weights"