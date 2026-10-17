import numpy as np
import pytest

from policy192 import (GLOBAL_BATCH, LR_END, LR_START, MICROBATCH, UPDATE_BUDGET, WORLD,
                       SmoothedSelection, is_held_out, learning_rate, market_probabilities)


def test_cosine_endpoints_and_monotonic_decay():
    values = [learning_rate(index) for index in range(UPDATE_BUDGET)]
    assert values[0] == pytest.approx(LR_START)
    assert values[-1] == pytest.approx(LR_END)
    assert all(before >= after > 0 for before, after in zip(values, values[1:]))
    assert learning_rate(-1) == values[0] and learning_rate(2000) == values[-1]


def test_batch_contract_per_arm():
    assert GLOBAL_BATCH % (WORLD * MICROBATCH) == 0
    assert WORLD == 32
    assert GLOBAL_BATCH // WORLD == 48
    assert GLOBAL_BATCH // (WORLD * MICROBATCH) == 24


def test_market_weights_renormalize_only_feasible_markets():
    weights = [10, 1000, 30]
    possible = [0, 2]
    probabilities = market_probabilities("B", possible, weights)
    np.testing.assert_allclose(probabilities, [0.25, 0.75])
    assert probabilities.sum() == pytest.approx(1)
    first = np.random.default_rng(np.random.SeedSequence([1337, 0, 12, 3]))
    second = np.random.default_rng(np.random.SeedSequence([1337, 0, 12, 3]))
    np.testing.assert_array_equal(first.choice(possible, 100, p=probabilities), second.choice(possible, 100, p=probabilities))
    assert market_probabilities("A", possible, weights) is None


def test_hash_holdout_is_stable_and_approximately_ten_percent():
    selected = [is_held_out("US", f"ticker-{index}") for index in range(10000)]
    assert 800 < sum(selected) < 1200
    assert selected == [is_held_out("US", f"ticker-{index}") for index in range(10000)]


def test_early_stop_requires_five_score_window_and_six_stale_validations():
    selection = SmoothedSelection(0.082)
    for _ in range(3):
        assert selection.observe(0.082)["smoothed"] is None
    for stale in range(1, 7):
        result = selection.observe(0.082)
        assert result["stale_validations"] == stale
        assert result["stop"] is (stale == 6)


def test_significant_smoothed_improvement_resets_patience():
    selection = SmoothedSelection(0.082)
    for _ in range(7):
        selection.observe(0.082)
    result = selection.observe(0.102)
    assert result["best"] and result["stale_validations"] == 0 and not result["stop"]


def test_selection_resume_preserves_window_and_patience():
    original = SmoothedSelection(0.08)
    for score in (0.07, 0.08, 0.09, 0.085, 0.08):
        original.observe(score)
    resumed = SmoothedSelection.from_state(original.state())
    assert resumed.state() == original.state()
    for score in (0.082, 0.083, 0.12):
        assert resumed.observe(score) == original.observe(score)


def test_resume_preserves_state_and_rejects_contract_changes(tmp_path, monkeypatch):
    import json
    import train192
    from ranker.data import write_json
    source, destination = tmp_path / "old/arm-A", tmp_path / "new/arm-A"
    checkpoint = source / "checkpoints/step_530"
    checkpoint.mkdir(parents=True)
    previous = {"experiment": "1.9.2", "arm": "A", "global_batch": 1536, "dp": 12}
    current = {**previous, "dp": 32}
    state = {"experiment": "1.9.2", "arm": "A", "world": 12, "step": 530, "new_updates": 202,
             "scheduled": 415232, "selection": SmoothedSelection(0.08).state()}
    write_json(checkpoint / "config.json", previous)
    write_json(checkpoint / "state.json", state)
    write_json(checkpoint / "manifest.json", {})
    for label in ("best", "latest"):
        write_json(source / f"{label}.json", {"directory": "checkpoints/step_530", "step": 530})
    monkeypatch.setattr(train192, "verify", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="frozen training contract"):
        train192.prepare_resume(source, destination, "A", {**current, "global_batch": 2048})
    result = train192.prepare_resume(source, destination, "A", current)
    assert result["state"] == {**state, "world": 32}
    assert json.loads((checkpoint / "state.json").read_text())["world"] == 12
    assert result["provenance"]["optimizer_preserved"] and result["provenance"]["sampler_preserved"]
    assert (destination / "checkpoints/step_530/state.json").read_bytes() == (checkpoint / "state.json").read_bytes()
    with pytest.raises(ValueError, match="already contains"):
        train192.prepare_resume(source, destination, "A", current)


def test_resume_restores_fp32_optimizer_and_learning_rate(tmp_path):
    import torch
    from types import SimpleNamespace
    from train192 import load_parent
    model = torch.nn.Linear(3, 1)
    model.config = SimpleNamespace(seed=1337)
    model.context = SimpleNamespace(dp_rank=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path / "recovery-00.pt")
    with torch.no_grad():
        model.weight.zero_()
    optimizer.state.clear()
    load_parent(tmp_path, model, optimizer, 202)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == learning_rate(202)
    assert all(value["step"].item() == 1 for value in optimizer.state.values())


def test_parent_combination_rank_roundtrip_and_repeat_detection():
    import math
    from ranker.sampler import coefficients, unrank
    from sampler192 import combination_rank, parent_used
    eligible = np.arange(12)
    total, key = math.comb(12, 3), 12345
    multiplier, offset = coefficients(1337, key, total)
    for index in range(total):
        rank = (multiplier * index + offset) % total
        tickers = unrank(12, 3, rank)
        assert combination_rank(12, tickers) == rank
        assert parent_used(eligible, tickers, key, 20, 1337) is (index < 20)


@pytest.mark.parametrize("arm", ["A", "B"])
def test_ten_thousand_draws_exclude_holdout_and_parent_baskets(tmp_path, arm):
    from ranker.contract import Task
    from ranker.sampler import Sampler
    from sampler192 import Sampler192

    class Panel:
        def __init__(self, market, excluded=()):
            self.market = market
            self.dates = np.arange(np.datetime64("2015-01-01"), np.datetime64("2016-08-23"))
            self.streaks = np.full((len(self.dates), 20), 600, dtype=np.uint16)
            self.selected = np.array([index for index in range(20) if index not in excluded])
            self.counts = np.full((57, len(self.dates)), len(self.selected), dtype=np.uint16)

        def eligible(self, cutoff, length):
            return self.selected

        def cutoffs(self, task, split):
            return np.arange(64, len(self.dates) - task.horizon)

    originals = [Panel("AU"), Panel("CA")]
    panels = [Panel("AU", (0, 3)), Panel("CA", (0, 3))]
    task = Task(64, 1, 3)
    parent = Sampler(originals, tmp_path / "parent.sqlite", fixed=task)
    old = {entry.identity for entry in parent.issue(100)}
    parent.save(tmp_path / "snapshot.sqlite")
    parent.close()
    sampler = Sampler192.from_parent(panels, originals, tmp_path / "snapshot.sqlite", tmp_path / "arm.sqlite",
                                    arm, [1, 3], fixed=task)
    assert sampler.issued == 100
    draws = sampler.issue(10000)
    assert sampler.issued == 10100
    assert all(not set(entry.tickers) & {0, 3} for entry in draws)
    identities = {entry.identity for entry in draws}
    assert len(identities) == 10000 and not identities & old
    assert sum(values["baskets"] for values in sampler.exposure().values()) == 10000
    sampler.save(tmp_path / "resume.sqlite")
    resumed = Sampler192.from_checkpoint(panels, originals, tmp_path / "resume.sqlite", tmp_path / "resumed.sqlite",
                                         arm, [1, 3], fixed=task)
    assert resumed.issued == sampler.issued
    assert resumed.exposure() == sampler.exposure()
    assert [entry.identity for entry in resumed.issue(48)] == [entry.identity for entry in sampler.issue(48)]
    resumed.close()
    sampler.close()


def test_hidden_model_maps_only_unknown_embedding_in_training_and_evaluation():
    from types import SimpleNamespace
    import torch
    from model192 import IdentityHiddenModel
    from ranking17.model import ModelConfig
    context = SimpleNamespace(device=torch.device("cpu"), tp_size=1, tp_rank=0, tp_group=None)
    config = ModelConfig(tickers=8, width=16, heads=2, feedforward=32, temporal_depth=1,
                         cross_depth=1, dropout=0, identity_dropout=0, rematerialize=False)
    model = IdentityHiddenModel(config, context)
    inputs = torch.randn(1, 3, 64)
    market, task = torch.tensor([0]), torch.tensor([[64, 1, 3]])
    first, second = torch.tensor([[0, 1, 2]]), torch.tensor([[3, 4, 5]])
    for training in (True, False):
        model.train(training)
        scores = model(inputs, first, market, task)[0]
        torch.testing.assert_close(scores, model(inputs, second, market, task)[0], rtol=0, atol=0)
        model.zero_grad()
        scores.sum().backward()
        assert torch.count_nonzero(model.ticker.grad[:-1]) == 0
        assert torch.count_nonzero(model.ticker.grad[-1]) > 0
    model.train()
    with pytest.raises(ValueError, match="must hide"):
        model(inputs, first, market, task, identity_masked=False)


def test_final_test_stays_sealed_before_explicit_flag_and_both_arms_finish(tmp_path):
    from evaluate192 import require_final_selection
    with pytest.raises(ValueError, match="sealed"):
        require_final_selection(tmp_path, False, {"final_test_unseen_user_confirmed": True})
    with pytest.raises(ValueError, match="Prior test exposure"):
        require_final_selection(tmp_path, True, {})
    with pytest.raises(FileNotFoundError):
        require_final_selection(tmp_path, True, {"final_test_unseen_user_confirmed": True})


def test_time_block_pairing_and_known_difference():
    from evaluate192 import time_block_difference
    first = [{"index": index, "market": "US", "task": [64, 1, 3], "cutoff": index * 90,
              "spearman": 0.08} for index in range(12)]
    second = [{**row, "spearman": 0.09} for row in first]
    result = time_block_difference(first, second)["US"]
    assert result["second_minus_first"] == pytest.approx(0.01)
    assert result["time_blocks"] == 12
    assert result["time_block_bootstrap_95pct"] == pytest.approx([0.01, 0.01])


def test_training_import_and_thirty_two_worker_distribution():
    from ranker.sampler import Assignment
    from train192 import distribute_assignments
    entries = [Assignment(0, index, (64, 1, 3), (0, 1, 2)) for index in range(GLOBAL_BATCH)]
    groups = distribute_assignments(entries, WORLD)
    assert len(groups) == 32
    assert all(len(rank) == 24 and sum(map(len, rank)) == 48 for rank in groups)
    assert sorted(entry.cutoff for rank in groups for batch in rank for entry in batch) == list(range(GLOBAL_BATCH))


def test_queued_and_incomplete_running_allocation_waits():
    from control192 import allocation
    job = {"Resource_List": {"nodect": 16, "walltime": "24:00:00"}, "job_state": "Q"}
    assert allocation(job) is None
    job["job_state"] = "R"
    assert allocation(job) is None
    job["stime"] = "Sat Oct 17 01:00:00 2026"
    assert allocation(job) is None
    job["exec_host"] = "+".join(f"x1000c1s{index}b0n0/0" for index in range(16))
    timing = allocation(job)
    assert len(timing["hosts"]) == 16
    assert timing["end_epoch"] - timing["start_epoch"] == 86400
    assert timing["end_epoch"] - timing["stop_epoch"] == 120


def test_native_batch_start_is_job_bound_and_duplicate_guarded(tmp_path, monkeypatch):
    import control192
    monkeypatch.setattr(control192, "JOB", "123.hunter-pbs01")
    monkeypatch.setattr(control192, "validate_source", lambda *args: None)
    monkeypatch.setattr(control192, "scheduler_job", lambda: {})
    monkeypatch.setattr(control192, "allocation", lambda job: {"end_epoch": 2000000000, "stop_epoch": 1999999880})
    monkeypatch.setattr(control192.time, "time", lambda: 1900000000)
    with pytest.raises(ValueError, match="native PBS"):
        control192.batch_start(tmp_path, tmp_path / "source")
    monkeypatch.setenv("PBS_JOBID", "123.hunter-pbs01")
    monkeypatch.setenv("PBS_ENVIRONMENT", "PBS_BATCH")
    assert control192.batch_start(tmp_path, tmp_path / "source") == 2000000000
    with pytest.raises(FileExistsError):
        control192.batch_start(tmp_path, tmp_path / "source")


def test_batch_failure_is_reported_immediately(tmp_path):
    import json
    import control192
    control192.batch_exit(tmp_path, 1)
    path = tmp_path / "experiment192/artifacts" / f"job-{control192.JOB}" / "control/failure.json"
    assert json.loads(path.read_text())["exit_code"] == 1


def test_public_logs_include_both_arms_and_exclude_private_files(tmp_path):
    from publish192 import JOB, collect
    for arm in ("A", "B"):
        root = tmp_path / "artifacts" / JOB / f"arm-{arm}"
        for name in (f"runs/arm-{arm}/events.out.tfevents.test", "history.jsonl", "ledgers/sampler.sqlite",
                     "private-evaluation/rows.json", "recovery-00.pt"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
    snapshot = collect(tmp_path, "metrics")
    assert all(f"runs/arm-{arm}/events.out.tfevents.test" in snapshot["files"] for arm in ("A", "B"))
    assert len(snapshot["files"]) == 4
    assert not any("private" in name or "sampler" in name or "recovery" in name for name in snapshot["files"])
    assert collect(tmp_path, "exports")["files"] == {}


def test_backup_cadence_starts_immediately_and_finishes_after_final_upload(tmp_path, monkeypatch):
    import publish192
    calls, waits = [], []
    def upload(*args):
        calls.append(args)
        return len(calls) == 2
    monkeypatch.setattr(publish192, "publish_once", upload)
    monkeypatch.setattr(publish192.time, "monotonic", lambda: 100)
    monkeypatch.setattr(publish192.threading.Event, "wait", lambda self, duration: waits.append(duration))
    publish192.relay(tmp_path, "http://local", "metrics")
    assert len(calls) == 2 and waits == [300]
    assert not (tmp_path / "metrics-relay.lock").exists()


def test_worker_shells_override_module_transport_after_environment_load():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "run.sh").read_text()
    assert "pbs_attach" not in source
    assert "--no-python" not in source
    assert "#PBS -l select=16:node_type=mi300a:ncpus=96:mpiprocs=1" in source
    assert "#PBS -r n" in source and "PBS_BATCH" in source
    assert "--nnodes=8 --nproc_per_node=4" in source
    assert "--nnodes=16 --nproc_per_node=4" in source
    shells = source.split("mpiexec -n 16")[1:]
    assert len(shells) == 2
    for shell in shells:
        assert shell.index("source ") < shell.index("export NCCL_NET=Socket") < shell.index("torch.distributed.run")


@pytest.mark.parametrize("node_rank", range(16))
def test_actual_worker_shell_assigns_two_disjoint_eight_node_arms(tmp_path, node_rank):
    import os
    from pathlib import Path
    import subprocess
    if os.name != "posix":
        pytest.skip("Execute the native Bash worker check on Hunter")
    environment_script = tmp_path / "source-six-node-20261014/env.sh"
    environment_script.parent.mkdir()
    environment_script.write_text("export NCCL_NET=AWS_Libfabric\n")
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$NCCL_NET" "$@"\n')
    executable.chmod(0o700)
    source = (Path(__file__).resolve().parents[1] / "run.sh").read_text()
    shell = source.split("/usr/bin/bash -c '\n", 1)[1].split("\n' 2>&1", 1)[0]
    environment = {**os.environ, "HUNTER_WORKSPACE": str(tmp_path), "PMI_RANK": str(node_rank),
        "MASTER_A": "node-a", "MASTER_B": "node-b", "EXPERIMENT192_ROOT": str(tmp_path / "run"),
        "TRAIN192_SOURCE": str(tmp_path), "PARENT192": "parent", "PREPARED192": "prepared", "TRAIN_END_AT": "2000000000",
        "RESUME192_ROOT": str(tmp_path / "previous")}
    result = subprocess.run(["bash", "-c", shell], env=environment, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert lines[0] == "Socket"
    assert "--nnodes=8" in lines and "--nproc_per_node=4" in lines
    assert f"--node_rank={node_rank % 8}" in lines
    assert f"--master_addr={'node-a' if node_rank < 8 else 'node-b'}" in lines
    assert lines[lines.index("--arm") + 1] == ("A" if node_rank < 8 else "B")
    assert lines[lines.index("--resume-arm-root") + 1] == str(tmp_path / "previous" / ("arm-A" if node_rank < 8 else "arm-B"))


def test_allocation_notification_retries_transient_disconnect(tmp_path, monkeypatch, capsys):
    import publish192
    from unittest.mock import Mock
    ready = Mock()
    ready.json.return_value = {"job": publish192.JOB, "experiment": "1.9.2", "state": "R", "time": 100}
    get = Mock(side_effect=[publish192.requests.ConnectionError("offline"), ready])
    waits = []
    monkeypatch.setattr(publish192.requests, "get", get)
    monkeypatch.setattr(publish192.time, "time", lambda: 100)
    monkeypatch.setattr(publish192.threading.Event, "wait", lambda self, duration: waits.append(duration))
    publish192.notify("http://local")
    assert get.call_count == 2 and waits == [30]
    assert "CURRENT_SESSION_NOTIFICATION" in capsys.readouterr().out


def test_allocation_notification_rejects_stale_heartbeat(monkeypatch):
    import publish192
    from unittest.mock import Mock
    response = Mock()
    response.json.return_value = {"job": publish192.JOB, "experiment": "1.9.2", "state": "Q", "time": 100}
    monkeypatch.setattr(publish192.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(publish192.time, "time", lambda: 281)
    with pytest.raises(RuntimeError, match="heartbeat"):
        publish192.notify("http://local")


def test_current_session_notification_never_invokes_cli(tmp_path, monkeypatch):
    import json
    import wake192
    from pathlib import Path
    from unittest.mock import Mock
    notify = Mock(return_value={"state": "R"})
    monkeypatch.setattr(wake192, "notify", notify)
    wake192.watch(tmp_path, "http://local")
    notify.assert_called_once_with("http://local", "allocation")
    assert json.loads((tmp_path / "allocation.json").read_text())["state"] == "notified"
    assert "subprocess" not in Path(wake192.__file__).read_text()
    assert not (tmp_path / "allocation.lock").exists()


def test_training_notification_requires_both_arms_checkpoints_and_updates(monkeypatch):
    import publish192
    from unittest.mock import Mock
    replies = []
    for updates in (0, 3):
        response = Mock()
        response.json.return_value = {"job": publish192.JOB, "experiment": "1.9.2", "state": "R", "time": 100,
            "arms": {arm: {"preflight_passed": True, "initial_checkpoint_complete": updates == 3,
                            "new_updates": updates} for arm in ("A", "B")}}
        replies.append(response)
    monkeypatch.setattr(publish192.requests, "get", Mock(side_effect=replies))
    monkeypatch.setattr(publish192.time, "time", lambda: 100)
    waits = []
    monkeypatch.setattr(publish192.threading.Event, "wait", lambda self, duration: waits.append(duration))
    assert publish192.notify("http://local", "training")["arms"]["A"]["new_updates"] == 3
    assert waits == [30]


def test_notification_records_failure_without_creating_agent(tmp_path, monkeypatch):
    import json
    import wake192
    from unittest.mock import Mock
    monkeypatch.setattr(wake192, "notify", Mock(side_effect=RuntimeError("batch failed")))
    with pytest.raises(RuntimeError, match="batch failed"):
        wake192.watch(tmp_path, "http://local", "training")
    assert json.loads((tmp_path / "training.json").read_text())["state"] == "failed"
    assert not (tmp_path / "training.lock").exists()


def test_export_relay_stops_when_allocation_ends_without_checkpoints(tmp_path, monkeypatch):
    import publish192
    from unittest.mock import Mock
    response = Mock()
    response.json.return_value = {"job": publish192.JOB, "experiment": "1.9.2", "finished": True}
    monkeypatch.setattr(publish192.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(publish192.requests, "post", Mock(return_value=Mock()))
    monkeypatch.setattr(publish192.shared, "snapshot_at", Mock(return_value={"experiment": "1.9.2", "frozen_at": "now", "files": {}}))
    assert publish192.publish_once(tmp_path, "http://local", "exports") is True