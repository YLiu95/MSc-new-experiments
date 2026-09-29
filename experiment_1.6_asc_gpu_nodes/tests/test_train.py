import pytest
import torch

from ranker.train import informative_count, learning_rate, optimizer_for, update
from ranker.model import ModelConfig, RankingModel


def test_global_informative_normalization_and_empty_update():
    torch.manual_seed(1337)
    config = ModelConfig(tickers=6, width=32, heads=4, feedforward=64, temporal_blocks=1,
                         cross_blocks=1, dropout=0, identity_dropout=0, activation_checkpointing=False)
    model = RankingModel(config)
    optimizer = optimizer_for(model, torch.device("cpu"))
    batch = {"inputs": torch.randn(4, 3, 64), "ticker_ids": torch.tensor([[0, 1, 2]] * 4),
             "market": torch.tensor([0, 1, 2, 3]), "task": torch.tensor([[64, 7, 3]] * 4),
             "labels": torch.tensor([[3., 2., 1.], [1., float("nan"), 1.],
                                     [0., 1., 2.], [1., 0., -1.]]),
             "observed": torch.tensor([[True, True, True], [True, False, True],
                                       [True, True, True], [True, True, True]]),
             "weight": torch.tensor([1., 1., 0., 1.])}
    assert informative_count(batch) == 2
    result = update(model, model, optimizer, [batch], torch.device("cpu"), 0, 1, True, 0)
    assert result["informative"] == 2 and result["pairs"] == 6
    assert result["loss"] > 0 and result["gradient_norm"] > 0
    batch["weight"] = torch.zeros(4)
    before = [parameter.clone() for parameter in model.parameters()]
    assert update(model, model, optimizer, [batch], torch.device("cpu"), 0, 1, True, 1)["skipped"]
    for parameter, old in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, old)


def test_schedule_is_one_continuous_30000_update_schedule():
    assert learning_rate(0) == pytest.approx(2e-7)
    assert learning_rate(999) == pytest.approx(2e-4)
    assert learning_rate(29999) == pytest.approx(4e-6)
    assert learning_rate(3000) < learning_rate(2999)