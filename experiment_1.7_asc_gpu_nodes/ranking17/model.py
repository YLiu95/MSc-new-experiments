from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

from ascgpu.model import (ColumnLinear, RowLinear, ParameterFactory, ParallelContext,
                         CopyToTensorGroup, GatherFromTensorGroup)


@dataclass
class ModelConfig:
    tickers: int = 24010
    width: int = 5120
    heads: int = 80
    feedforward: int = 20480
    temporal_depth: int = 20
    cross_depth: int = 12
    dropout: float = 0.15
    identity_dropout: float = 0.10
    seed: int = 1337
    rematerialize: bool = True


class Block(nn.Module):
    def __init__(self, config, factory):
        super().__init__()
        self.context = factory.context
        self.heads = config.heads // self.context.tp_size
        self.head_width = config.width // config.heads
        self.dropout = config.dropout
        local_width = config.width // self.context.tp_size
        self.qkv_weight = factory.parameter((3 * local_width, config.width), "qkv", scale=config.width ** -0.5)
        self.qkv_bias = factory.parameter((3 * local_width,), "qkv", zeros=True)
        self.output = RowLinear(config.width, config.width, factory)
        self.norm_attention = nn.LayerNorm(config.width, device=self.context.device)
        self.norm_feedforward = nn.LayerNorm(config.width, device=self.context.device)
        self.expand = ColumnLinear(config.width, config.feedforward, factory)
        self.contract = RowLinear(config.feedforward, config.width, factory)

    def forward(self, states):
        batch, length, width = states.shape
        copied = CopyToTensorGroup.apply(self.norm_attention(states), self.context.tp_group, self.context.tp_size)
        projected = functional.linear(copied, self.qkv_weight, self.qkv_bias)
        query, key, value = projected.reshape(batch, length, 3, self.heads, self.head_width).unbind(2)
        attended = functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, -1)
        states = states + functional.dropout(self.output(attended), self.dropout, self.training)
        hidden = functional.gelu(self.expand(self.norm_feedforward(states)))
        hidden = functional.dropout(hidden, self.dropout, self.training)
        return states + functional.dropout(self.contract(hidden), self.dropout, self.training)


class RankingModel(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        if config.width % config.heads or config.heads % context.tp_size or config.feedforward % context.tp_size:
            raise ValueError("Model dimensions must divide the TP mesh")
        self.config, self.context = config, context
        factory = ParameterFactory(context, config.seed)
        local_width = config.width // context.tp_size
        self.patch = ColumnLinear(8, config.width, factory)
        self.positions = factory.parameter((64, local_width), 1)
        self.ticker = factory.parameter((config.tickers + 1, local_width), 1)
        self.market = factory.parameter((13, local_width), 1)
        self.temporal = nn.ModuleList(Block(config, factory) for _ in range(config.temporal_depth))
        self.base_norm = nn.LayerNorm(config.width, device=context.device)
        self.length_embedding = factory.parameter((57, 16))
        self.horizon_embedding = factory.parameter((90, 16))
        self.basket_embedding = factory.parameter((126, 16))
        self.conditioner_weight = factory.parameter((128, 48), scale=48 ** -0.5)
        self.conditioner_bias = factory.parameter((128,), zeros=True)
        self.film_weight = factory.parameter((2 * local_width, 128), "qkv", zeros=True)
        self.film_bias = factory.parameter((2 * local_width,), "qkv", zeros=True)
        self.cross = nn.ModuleList(Block(config, factory) for _ in range(config.cross_depth))
        self.head_norm = nn.LayerNorm(config.width, device=context.device)
        self.head_hidden = ColumnLinear(config.width, 128, factory)
        self.head_output = RowLinear(128, 1, factory)

    def gather(self, values):
        return GatherFromTensorGroup.apply(values, self.context.tp_group, self.context.tp_size, self.context.tp_rank)

    def stack(self, states, blocks):
        for block in blocks:
            states = checkpoint(block, states, use_reentrant=False) if self.training and self.config.rematerialize else block(states)
        return states

    def forward(self, inputs, ticker_ids, market_ids, tasks, identity_masked=False):
        batch, basket, length = inputs.shape
        if length % 8 or length > 512 or ticker_ids.shape != (batch, basket):
            raise ValueError("Expected complete eight-return patches")
        identities = ticker_ids
        if identity_masked:
            identities = torch.full_like(identities, self.config.tickers)
        elif self.training and self.config.identity_dropout:
            identities = torch.where(torch.rand(identities.shape, device=inputs.device) < self.config.identity_dropout,
                                     self.config.tickers, identities)
        positions = torch.arange(length // 8 - 1, -1, -1, device=inputs.device)
        states = self.patch(inputs.reshape(batch, basket, length // 8, 8))
        states = states + functional.embedding(positions, self.positions).to(states.dtype)
        states = states + functional.embedding(identities, self.ticker)[:, :, None].to(states.dtype)
        states = states + functional.embedding(market_ids, self.market)[:, None, None].to(states.dtype)
        states = self.gather(states).reshape(batch * basket, length // 8, self.config.width)
        states = self.stack(states, self.temporal)
        base = self.base_norm(states.mean(1)).reshape(batch, basket, self.config.width)
        features = torch.cat((functional.embedding((tasks[:, 0] - 64) // 8, self.length_embedding),
                              functional.embedding(tasks[:, 1] - 1, self.horizon_embedding),
                              functional.embedding(tasks[:, 2] - 3, self.basket_embedding)), -1)
        hidden = functional.gelu(functional.linear(features, self.conditioner_weight, self.conditioner_bias))
        hidden = CopyToTensorGroup.apply(hidden, self.context.tp_group, self.context.tp_size)
        gamma, beta = functional.linear(hidden, self.film_weight, self.film_bias).chunk(2, -1)
        contextual = base * (1 + self.gather(gamma)[:, None]) + self.gather(beta)[:, None]
        contextual = self.stack(contextual, self.cross)
        hidden = functional.gelu(self.head_hidden(self.head_norm(contextual)))
        hidden = functional.dropout(hidden, self.config.dropout, self.training)
        return self.head_output(hidden).squeeze(-1).float(), base, contextual

    def parameter_count(self):
        return sum(parameter.numel() * (self.context.tp_size if getattr(parameter, "tp_kind", None) is not None else 1)
                   for parameter in self.parameters())