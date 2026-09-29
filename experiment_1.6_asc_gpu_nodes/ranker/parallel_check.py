from __future__ import annotations

import argparse
import json

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .losses import pairwise_logistic
from .model import ModelConfig, RankingModel


def check(device: torch.device) -> dict:
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError("The registered uneven-label diagnostic uses exactly two DDP workers")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = ModelConfig(tickers=12, width=32, heads=4, feedforward=64, temporal_blocks=1,
                         cross_blocks=1, dropout=0, identity_dropout=0, activation_checkpointing=False)
    torch.manual_seed(1337)
    model = RankingModel(config).to(device).train()
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    generator = torch.Generator().manual_seed(7331)
    inputs = torch.randn(4, 3, 64, generator=generator).to(device)
    tickers = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [7, 8, 9]], device=device)
    markets = torch.tensor([0, 1, 2, 3], device=device)
    tasks = torch.tensor([[64, 7, 3]] * 4, device=device)
    labels = torch.tensor([[3., 2., 1.], [2., 1., float("nan")],
                           [2., 2., 2.], [3., 2., 1.]], device=device)
    observed = torch.isfinite(labels)
    weights = torch.tensor([1., 1., 1., 0.], device=device)
    indices = slice(rank * 2, (rank + 1) * 2)
    local_informative = torch.tensor([2 if rank == 0 else 0], dtype=torch.long, device=device)
    dist.all_reduce(local_informative)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.enable_grad()
    with context:
        scores = wrapped(inputs[indices], tickers[indices], markets[indices], tasks[indices])[0]
        result = pairwise_logistic(scores, labels[indices], observed[indices], weights[indices])
        (result.total * 2 / local_informative.item()).backward()
    distributed_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    distributed_scores = scores.detach().float()
    torch.manual_seed(1337)
    reference = RankingModel(config).to(device).train()
    with context:
        target_scores = reference(inputs, tickers, markets, tasks)[0]
        (pairwise_logistic(target_scores, labels, observed, weights).total / 2).backward()
    tolerance = {"atol": 0.02, "rtol": 0.02} if device.type == "cuda" else {"atol": 1e-6, "rtol": 1e-5}
    errors = []
    for trained, baseline in zip(distributed_gradients, reference.parameters()):
        torch.testing.assert_close(trained.float(), baseline.grad.float(), **tolerance)
        errors.append(float((trained.float() - baseline.grad.float()).abs().max().item()))
    torch.testing.assert_close(distributed_scores, target_scores[indices].detach().float(), **tolerance)
    result = {"backend": "bf16-cuda" if device.type == "cuda" else "fp32-cpu",
              "rank": rank, "informative_on_rank": 2 if rank == 0 else 0, "padding_on_rank": int(rank == 1),
              "maximum_gradient_absolute_error": max(errors), "tolerances": tolerance,
              "score_maximum_absolute_error": float((distributed_scores - target_scores[indices].detach()).abs().max().item())}
    received = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(result, received, dst=0)
    dist.barrier()
    dist.destroy_process_group()
    return {"workers": received} if rank == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda", int(__import__("os").environ.get("LOCAL_RANK", "0"))) if arguments.device == "cuda" else torch.device("cpu")
    result = check(device)
    if result is not None:
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()