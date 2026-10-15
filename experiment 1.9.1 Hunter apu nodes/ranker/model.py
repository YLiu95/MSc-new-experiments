from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    tickers: int
    width: int = 256
    heads: int = 8
    feedforward: int = 1024
    temporal_blocks: int = 4
    cross_blocks: int = 2
    dropout: float = 0.15
    identity_dropout: float = 0.10
    activation_checkpointing: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.attention = nn.MultiheadAttention(config.width, config.heads, dropout=config.dropout, batch_first=True, bias=True)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.feedforward = nn.Sequential(nn.Linear(config.width, config.feedforward), nn.GELU(),
                                         nn.Dropout(config.dropout), nn.Linear(config.feedforward, config.width),
                                         nn.Dropout(config.dropout))

    def forward(self, states: torch.Tensor, padding: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.attention_norm(states)
        attended = self.attention(normalized, normalized, normalized, key_padding_mask=padding, need_weights=False)[0]
        states = states + self.attention_dropout(attended)
        return states + self.feedforward(self.feedforward_norm(states))


class RankingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.patch = nn.Linear(8, config.width)
        self.positions = nn.Embedding(64, config.width)
        self.ticker = nn.Embedding(config.tickers + 1, config.width)
        self.market = nn.Embedding(13, config.width)
        self.temporal = nn.ModuleList(TransformerBlock(config) for _ in range(config.temporal_blocks))
        self.base_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.length_embedding = nn.Embedding(57, 16)
        self.horizon_embedding = nn.Embedding(90, 16)
        self.basket_embedding = nn.Embedding(126, 16)
        self.conditioner = nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Linear(128, config.width * 2))
        self.cross = nn.ModuleList(TransformerBlock(config) for _ in range(config.cross_blocks))
        self.head = nn.Sequential(nn.LayerNorm(config.width, eps=1e-5), nn.Linear(config.width, 128),
                                  nn.GELU(), nn.Dropout(config.dropout), nn.Linear(128, 1))
        self.apply(self._initialize)
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def _block(self, block: TransformerBlock, states: torch.Tensor, padding: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and self.config.activation_checkpointing:
            return checkpoint(block, states, padding, use_reentrant=False)
        return block(states, padding)

    def forward(self, inputs: torch.Tensor, ticker_ids: torch.Tensor, market_ids: torch.Tensor,
                tasks: torch.Tensor, patch_mask: torch.Tensor | None = None,
                identity_masked: bool = False, conditioned: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, basket, length = inputs.shape
        if length % 8 or length // 8 > 64 or ticker_ids.shape != (batch, basket):
            raise ValueError("Expected complete eight-return patches and one ID per ticker")
        patches = inputs.reshape(batch, basket, length // 8, 8)
        valid = torch.ones(patches.shape[:-1], dtype=torch.bool, device=inputs.device) if patch_mask is None else patch_mask
        if valid.shape != patches.shape[:-1] or not torch.all(valid.any(dim=-1)):
            raise ValueError("Each ticker must contain at least one valid patch")
        positions = valid.flip(-1).to(torch.long).cumsum(dim=-1).flip(-1).sub(1).clamp(min=0)
        identities = ticker_ids
        if identity_masked:
            identities = torch.full_like(identities, self.config.tickers)
        elif self.training and self.config.identity_dropout:
            identities = torch.where(torch.rand(identities.shape, device=identities.device) < self.config.identity_dropout,
                                     self.config.tickers, identities)
        states = (self.patch(patches) + self.positions(positions) + self.ticker(identities)[:, :, None]
                  + self.market(market_ids)[:, None, None])
        states = states.reshape(batch * basket, length // 8, self.config.width)
        padding = ~valid.reshape(batch * basket, length // 8)
        for block in self.temporal:
            states = self._block(block, states, padding)
        states = states.reshape(batch, basket, length // 8, self.config.width)
        base = self.base_norm((states * valid[..., None]).sum(dim=2) / valid.sum(dim=2).unsqueeze(-1))
        if conditioned:
            features = torch.cat((self.length_embedding((tasks[:, 0] - 64) // 8),
                                  self.horizon_embedding(tasks[:, 1] - 1),
                                  self.basket_embedding(tasks[:, 2] - 3)), dim=-1)
            gamma, beta = self.conditioner(features).chunk(2, dim=-1)
            contextual = base * (1 + gamma[:, None]) + beta[:, None]
        else:
            contextual = base
        for block in self.cross:
            contextual = self._block(block, contextual)
        scores = self.head(contextual).squeeze(-1)
        return scores, base, contextual