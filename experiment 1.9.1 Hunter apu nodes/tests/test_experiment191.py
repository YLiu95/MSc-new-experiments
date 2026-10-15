import pytest

from train191 import GLOBAL_BATCH, LR_CAP, MICROBATCH, WORLD, capped_learning_rate, distribute_assignments, migrated_state, next_action


def test_rate_cap_applies_after_parent_step_and_through_schedule():
    assert capped_learning_rate(328) == LR_CAP
    assert all(0 < capped_learning_rate(step) <= LR_CAP for step in range(40000))
    assert capped_learning_rate(0) < LR_CAP


def test_all_48_workers_get_equal_shape_matched_microsteps_without_extra_samples():
    from types import SimpleNamespace
    entries = [SimpleNamespace(position=position, task=(16 * (1 + (position // 4) % 32), 1, 32))
               for position in range(GLOBAL_BATCH)]
    assignments = distribute_assignments(entries)
    assert len(assignments) == WORLD
    assert {len(groups) for groups in assignments} == {16}
    assert all(len(group) == MICROBATCH and len({entry.task for entry in group}) == 1
               for groups in assignments for group in groups)
    assert sorted(entry.position for groups in assignments for group in groups for entry in group) == list(range(GLOBAL_BATCH))
    loads = [sum(group[0].task[0] * group[0].task[2] for group in groups) for groups in assignments]
    assert max(loads) / min(loads) < 1.1


def test_final_save_finishes_before_two_minute_reserve():
    stop_at = 1000 - 120
    assert next_action(100, 300, stop_at, 100) == "update"
    assert next_action(300, 300, stop_at, 100) == "validate"
    assert next_action(700, 600, stop_at, 100) == "update"
    assert next_action(770, 900, stop_at, 100) == "finish"
    assert next_action(stop_at, 900, stop_at, 100) == "finish"


def test_migration_preserves_progress_and_rejects_wrong_parent():
    config = {"tp": 1, "dp": 8, "global_batch": 320, "parameters": 1069302417}
    previous = {"world": 8, "step": 328, "scheduled": 104960, "best_metric": 0.08534306347402562}
    state = migrated_state(config, previous)
    assert state["world"] == 48 and state["scheduled"] == previous["scheduled"]
    assert state["best_metric"] == previous["best_metric"]
    assert previous["world"] == 8
    with pytest.raises(ValueError):
        migrated_state(config, {**previous, "step": 2172})


def test_queue_watcher_uses_existing_job_and_original_deadlines(tmp_path):
    from control191 import JOB, allocation, launch_command
    job = {"job_state": "Q", "Resource_List": {"nodect": 12, "walltime": "07:00:00"}}
    assert allocation(job) is None
    job.update(job_state="R", stime="Fri Oct 16 00:00:00 2026",
               exec_host="+".join(f"x1000c0s{index}b0n0/0" for index in range(12)))
    timing = allocation(job)
    assert timing["end_epoch"] - timing["start_epoch"] == 25200
    assert timing["end_epoch"] - timing["stop_epoch"] == 120
    command = launch_command(tmp_path, tmp_path / "source", tmp_path / "run", timing)
    assert JOB in command and "pbs_attach" in command and "nohup" in command
    assert "qsub" not in command and "qdel" not in command
    with pytest.raises(ValueError):
        allocation({**job, "exec_host": "not-our-node/0"})


def test_update_enforces_cap_at_optimizer_step(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    import train191

    class Toy(torch.nn.Linear):
        def forward(self, inputs, ticker_ids, market, task):
            return (super().forward(inputs),)

    model = Toy(2, 2)
    model.context = SimpleNamespace(device=torch.device("cpu"), dp_group=None, dp_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = {"inputs": torch.ones(1, 2), "ticker_ids": None, "market": None, "task": None,
             "labels": torch.tensor([[1.0, 2.0]]), "observed": torch.ones(1, 2, dtype=torch.bool),
             "weight": torch.ones(1)}
    monkeypatch.setattr(train191.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(train191.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(train191.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(train191, "clip_global_norm", lambda model, limit: float(torch.nn.utils.clip_grad_norm_(model.parameters(), limit)))
    observed = []
    optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: observed.append(optimizer.param_groups[0]["lr"]))
    train191.update(model, model, optimizer, [batch], 328)
    train191.update(model, model, optimizer, [batch], 2000)
    assert observed == [LR_CAP, LR_CAP]


def test_parent_loader_preserves_weights_and_adam(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    import torch
    from train191 import load_parent
    model = torch.nn.Linear(2, 1)
    model.config = SimpleNamespace(seed=1337)
    model.context = SimpleNamespace(dp_rank=47)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected = deepcopy(model.state_dict())
    moments = deepcopy(optimizer.state_dict()["state"])
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path / "recovery-00.pt")
    optimizer.state.clear()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    load_parent(tmp_path, model, optimizer)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in expected.items())
    assert all(torch.equal(value, optimizer.state_dict()["state"][index][name])
               for index, state in moments.items() for name, value in state.items())
    assert optimizer.param_groups[0]["lr"] == LR_CAP


def test_public_and_metrics_snapshots_exclude_recovery(tmp_path):
    import json
    from publish191 import JOB, collect, status
    from ranker.data import write_json
    from ranking17.recovery import digest
    root = tmp_path / "artifacts" / JOB
    directory = root / "exports" / "step_00000328"
    directory.mkdir(parents=True)
    for name, content in {"weights-00.safetensors": b"weights", "config.json": b"{}", "state.json": b'{"step":328}'}.items():
        (directory / name).write_bytes(content)
    write_json(directory / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                                             for path in directory.iterdir()})
    (directory / "COMPLETE").write_text(digest(directory / "manifest.json"))
    for label in ("best", "latest"):
        write_json(root / "exports" / f"{label}.json", {"directory": directory.name})
    (root / "recovery-00.pt").write_bytes(b"private")
    public, metrics = collect(tmp_path, "public"), collect(tmp_path, "metrics")
    assert "best/weights-00.safetensors" in public["files"]
    assert not any(name.startswith(("best/", "latest/")) for name in metrics["files"])
    assert not any("recovery-" in name or "rng-" in name or "sampler.sqlite" in name for name in public["files"])
    assert status(tmp_path)["latest_step"] == 328
    with pytest.raises(ValueError, match="stopped training"):
        collect(tmp_path, "private")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["recovery-00.pt"] = {"bytes": 0, "sha256": "bad"}
    write_json(directory / "manifest.json", manifest)
    (directory / "COMPLETE").write_text(digest(directory / "manifest.json"))
    with pytest.raises(ValueError, match="Private file"):
        collect(tmp_path, "public")


def test_model_card_metadata_is_valid():
    from pathlib import Path
    from huggingface_hub import ModelCard
    card = ModelCard.load(str(Path(__file__).parents[1] / "MODEL_CARD.md"))
    assert "tensorboard" in card.data.tags