from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as functional


class RankingLoss(NamedTuple):
    total: torch.Tensor
    informative: torch.Tensor
    pairs: torch.Tensor
    per_sample: torch.Tensor


def pairwise_logistic(scores: torch.Tensor, labels: torch.Tensor, observed: torch.Tensor,
                      sample_weight: torch.Tensor | None = None, temperature: float = 1.0) -> RankingLoss:
    if scores.shape != labels.shape or scores.shape != observed.shape or temperature <= 0:
        raise ValueError("Invalid ranking batch or temperature")
    labels = torch.nan_to_num(labels.float())
    scores = scores.float()
    count = scores.shape[1]
    upper = torch.triu(torch.ones((count, count), device=scores.device, dtype=torch.bool), diagonal=1)
    differences = labels[:, :, None] - labels[:, None, :]
    pairs = upper & observed[:, :, None] & observed[:, None, :] & (differences != 0)
    margin = torch.sign(differences) * (scores[:, :, None] - scores[:, None, :]) / temperature
    losses = functional.softplus(-margin).masked_fill(~pairs, 0)
    pair_counts = pairs.sum(dim=(1, 2))
    per_sample = losses.sum(dim=(1, 2)) / pair_counts.clamp(min=1)
    weights = torch.ones_like(per_sample) if sample_weight is None else sample_weight.float()
    informative = (pair_counts > 0) & (weights > 0)
    return RankingLoss((per_sample * weights).sum(), informative.sum(), (pair_counts * informative).sum(), per_sample)