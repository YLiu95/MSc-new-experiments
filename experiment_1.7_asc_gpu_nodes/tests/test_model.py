import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ranking17.model import ModelConfig, ParallelContext, RankingModel


def test_count_and_forward():
    with FakeTensorMode():
        full = RankingModel(ModelConfig(), ParallelContext(tp_size=8))
        assert full.parameter_count() > 10_000_000_000
        print("FULL_PARAMETER_COUNT", full.parameter_count())
    model = RankingModel(ModelConfig(tickers=20, width=32, heads=8, feedforward=64,
                                     temporal_depth=1, cross_depth=1), ParallelContext())
    scores, base, contextual = model(torch.randn(1, 3, 64), torch.tensor([[0, 1, 2]]),
                                     torch.tensor([0]), torch.tensor([[64, 7, 3]]))
    assert scores.shape == (1, 3)
    scores.square().sum().backward()
    assert model.film_weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())