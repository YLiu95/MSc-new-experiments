---
language: en
tags:
- pytorch
- safetensors
- tensorboard
- finance
- time-series
- ranking
base_model: YL95/experiment-1.7.1-hunter-apu-nodes
---

# Experiment 1.9: 1.069B Depth-Pruned Hunter Ranker

This model has 1,069,302,417 parameters. It retains temporal blocks 0 and 19,
cross-sectional block 11, embeddings, conditioning, norms and prediction head
from Experiment 1.7.1's best checkpoint at step 305. Indices are zero-based.
Width 5120, 80 attention heads and feedforward width 20480 are unchanged.
Layer selection was structural, not importance-ranked or an optimized search.

Parent revision: `586d6a4eba7f233b1070fc59acfef8857aad6444`.
TP4 parent shards were merged into a TP1 child, preserving packed Q/K/V and
FiLM component ordering. Adam, RNG, sampler and learning-rate warmup reset;
earlier training samples may recur. Training used four data-parallel MI300As
on one Hunter node, global batch 320, microbatch 1, BF16 compute, FP32 weights
and Adam, activation checkpointing, and the original dropout/data contract.

## Results

The run completed 66 updates and 21,120 scheduled samples. Best and latest
saved checkpoints both correspond to step 66. Monitoring Spearman improved
from 0.06407690096409865 before training to 0.08075104483874465 at step 66.
The parent scored 0.08394159463901718 on the same monitoring subset.
These are 1,616-basket monitoring results, not held-out or full-test results.
The last training loss was 0.6909430596977473.

The scheduler recorded a termination request from the owning account near
15:00 CEST on 2026-10-15, before the scheduled 16:11 CEST walltime endpoint.
Training logged a termination signal after completing and verifying the
step-66 checkpoint. This was not a full two-hour training run.

## Contents

- `best/` and `latest/`: BF16 inference matrices, FP32 vectors, configuration
  and saved-step metadata. Both labels refer to the same step in this run.
- `runs/`: original TensorBoard event files, visible in the hosted Metrics tab.
- `reports/`: aggregate training/validation history, hardware preflight,
  checkpoint verification and scheduler status.
- `source/`: exact deployed training archive and reused runtime source archive.

Full FP32/Adam/RNG/sampler recovery checkpoints and private data are not public
uploads; they remain in the temporary Hunter workspace. These inference
exports alone cannot exactly resume training. No bitwise-resume certification
is claimed for Experiment 1.9.

Source: [Experiment 1.9 on GitHub](https://github.com/YLiu95/MSc-new-experiments/tree/main/experiment%201.9%20Hunter%20apu%20nodes).
Metrics: [Hosted TensorBoard](https://huggingface.co/YL95/experiment-1.9-hunter-apu-nodes/tensorboard).

This is research software for cross-sectional ranking, not a general-purpose
language model or a validated trading system. It does not establish future
investment performance and is not financial advice.