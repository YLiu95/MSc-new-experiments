import torch

from ranker.model import ModelConfig, RankingModel
from ranker.profile import step, synthetic
from ranker.train import optimizer_for


def test_small_and_worst_shapes_share_one_finite_step():
    config = ModelConfig(tickers=140, width=32, heads=4, feedforward=64,
                         temporal_blocks=1, cross_blocks=1, dropout=0)
    model = RankingModel(config)
    optimizer = optimizer_for(model, torch.device("cpu"))
    for length, horizon, basket in ((64, 1, 3), (512, 90, 128)):
        batch = synthetic(config, torch.device("cpu"), 1, length, horizon, basket)
        norm, variance = step(model, model, optimizer, batch, 1, torch.device("cpu"), 1)
        assert norm > 0 and variance > 0