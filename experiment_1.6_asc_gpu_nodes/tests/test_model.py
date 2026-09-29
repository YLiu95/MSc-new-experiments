import torch

from ranker.losses import pairwise_logistic
from ranker.model import ModelConfig, RankingModel


def test_loss_normalizes_per_sample_and_skips_missing_tied_or_padding():
    scores = torch.tensor([[2., 1., 0.], [1., 1., 1.], [0., 0., 0.]], requires_grad=True)
    labels = torch.tensor([[3., 2., 1.], [4., 4., float("nan")], [1., 0., -1.]])
    observed = torch.isfinite(labels)
    result = pairwise_logistic(scores, labels, observed, torch.tensor([1., 1., 0.]))
    expected = (2 * torch.nn.functional.softplus(torch.tensor(-1.)) + torch.nn.functional.softplus(torch.tensor(-2.))) / 3
    torch.testing.assert_close(result.total, expected)
    assert (result.informative.item(), result.pairs.item()) == (1, 3)
    result.total.backward()
    assert scores.grad[0].abs().sum() > 0
    assert scores.grad[1:].abs().sum() == 0
    empty = pairwise_logistic(scores.detach().requires_grad_(), labels, observed, torch.zeros(3))
    assert empty.total.item() == 0 and empty.informative.item() == 0


def test_equivariance_padding_and_base_embedding_contract():
    torch.manual_seed(1337)
    config = ModelConfig(tickers=12, width=32, heads=4, feedforward=64, temporal_blocks=1,
                         cross_blocks=1, dropout=0, identity_dropout=0, activation_checkpointing=False)
    model = RankingModel(config).eval()
    inputs = torch.randn(2, 3, 64)
    identities = torch.tensor([[0, 1, 2], [3, 4, 5]])
    markets = torch.tensor([0, 1])
    tasks = torch.tensor([[64, 1, 3], [64, 7, 3]])
    with torch.no_grad():
        scores, base, context = model(inputs, identities, markets, tasks)
        permutation = torch.tensor([2, 0, 1])
        other_scores, other_base, other_context = model(inputs[:, permutation], identities[:, permutation], markets, tasks)
        torch.testing.assert_close(other_scores, scores[:, permutation], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(other_base, base[:, permutation], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(other_context, context[:, permutation], atol=1e-6, rtol=1e-6)
        padded = torch.cat((torch.zeros(2, 3, 8), inputs), dim=-1)
        mask = torch.ones(2, 3, 9, dtype=torch.bool)
        mask[:, :, 0] = False
        padded_scores, padded_base, _ = model(padded, identities, markets, tasks, patch_mask=mask)
        torch.testing.assert_close(padded_scores, scores, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(padded_base, base, atol=1e-5, rtol=1e-5)
        shifted = tasks.clone()
        shifted[:, 1] += 1
        _, shifted_base, _ = model(inputs, identities, markets, shifted)
        torch.testing.assert_close(shifted_base, base)
        model.conditioner[-1].weight[0, 16] = 0.5
        _, same_base, changed_context = model(inputs, identities, markets, shifted)
        torch.testing.assert_close(same_base, base)
        assert not torch.allclose(context, changed_context)


def test_ranking_gradients_reach_shared_path_and_unknown_identity():
    torch.manual_seed(3)
    config = ModelConfig(tickers=8, width=32, heads=4, feedforward=64, temporal_blocks=1,
                         cross_blocks=1, dropout=0, identity_dropout=1, activation_checkpointing=True)
    model = RankingModel(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.randn(2, 3, 64)
    identities = torch.tensor([[0, 1, 2], [3, 4, 5]])
    markets = torch.tensor([0, 1])
    tasks = torch.tensor([[64, 1, 3], [64, 7, 3]])
    labels = torch.tensor([[3., 1., -1.], [-2., 2., 0.]])
    for _ in range(2):
        optimizer.zero_grad()
        scores, _, _ = model(inputs, identities, markets, tasks)
        pairwise_logistic(scores, labels, torch.ones_like(labels, dtype=torch.bool)).total.backward()
        assert model.patch.weight.grad.abs().sum() > 0
        assert model.ticker.weight.grad[-1].abs().sum() > 0
        assert model.market.weight.grad.abs().sum() > 0
        assert model.cross[0].attention.in_proj_weight.grad.abs().sum() > 0
        assert model.conditioner[-1].weight.grad.abs().sum() > 0
        optimizer.step()
    optimizer.zero_grad()
    scores, _, _ = model(inputs, identities, markets, tasks)
    pairwise_logistic(scores, labels, torch.ones_like(labels, dtype=torch.bool)).total.backward()
    assert model.horizon_embedding.weight.grad.abs().sum() > 0


def test_tiny_repeated_basket_is_learnable_without_score_collapse():
    torch.manual_seed(1337)
    config = ModelConfig(tickers=6, width=32, heads=4, feedforward=64,
                         temporal_blocks=1, cross_blocks=1, dropout=0,
                         identity_dropout=0, activation_checkpointing=False)
    model = RankingModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    inputs = torch.randn(2, 3, 64)
    tickers = torch.tensor([[0, 1, 2], [2, 3, 4]])
    markets = torch.tensor([0, 0])
    tasks = torch.tensor([[64, 7, 3], [64, 7, 3]])
    labels = torch.tensor([[2., 1., 0.], [0., 2., 1.]])
    observed = torch.ones_like(labels, dtype=torch.bool)
    losses = []
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        scores = model(inputs, tickers, markets, tasks)[0]
        loss = pairwise_logistic(scores, labels, observed).total / len(inputs)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < 0.8 * losses[0]
    assert scores.detach().float().var(unbiased=False) > 0