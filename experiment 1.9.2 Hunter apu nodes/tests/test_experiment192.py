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
    assert GLOBAL_BATCH // WORLD == 64
    assert GLOBAL_BATCH // (WORLD * MICROBATCH) == 32


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


def test_training_import_and_sixteen_worker_distribution():
    from ranker.sampler import Assignment
    from train192 import distribute_assignments
    entries = [Assignment(0, index, (64, 1, 3), (0, 1, 2)) for index in range(GLOBAL_BATCH)]
    groups = distribute_assignments(entries, WORLD)
    assert len(groups) == 24
    assert all(len(rank) == 32 and sum(map(len, rank)) == 64 for rank in groups)
    assert sorted(entry.cutoff for rank in groups for batch in rank for entry in batch) == list(range(GLOBAL_BATCH))


def test_queued_and_incomplete_running_allocation_waits():
    from control192 import allocation
    job = {"Resource_List": {"nodect": 12, "walltime": "24:00:00"}, "job_state": "Q"}
    assert allocation(job) is None
    job["job_state"] = "R"
    assert allocation(job) is None
    job["stime"] = "Sat Oct 17 01:00:00 2026"
    assert allocation(job) is None
    job["exec_host"] = "+".join(f"x1000c1s{index}b0n0/0" for index in range(12))
    timing = allocation(job)
    assert len(timing["hosts"]) == 12
    assert timing["end_epoch"] - timing["start_epoch"] == 86400
    assert timing["end_epoch"] - timing["stop_epoch"] == 120


def test_launch_is_job_bound_and_duplicate_guarded(tmp_path):
    from control192 import launch_command
    command = launch_command(tmp_path, tmp_path / "source", tmp_path / "run", {"end_epoch": 2000000000})
    assert "start.lock" in command and "|| exit 0" in command
    assert "pbs_attach -j 287925.hunter-pbs01 -s" in command
    assert "TRAIN_END_AT=2000000000" in command
    assert "qsub" not in command and "qdel" not in command


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
    shells = source.split("mpiexec -n 12")[1:]
    assert len(shells) == 2
    for shell in shells:
        assert shell.index("source ") < shell.index("export NCCL_NET=Socket") < shell.index("torch.distributed.run")


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
    assert "ALLOCATION_NOTIFICATION" in capsys.readouterr().out


def test_allocation_notification_rejects_stale_heartbeat(monkeypatch):
    import publish192
    from unittest.mock import Mock
    response = Mock()
    response.json.return_value = {"job": publish192.JOB, "experiment": "1.9.2", "state": "Q", "time": 100}
    monkeypatch.setattr(publish192.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(publish192.time, "time", lambda: 281)
    with pytest.raises(RuntimeError, match="heartbeat"):
        publish192.notify("http://local")