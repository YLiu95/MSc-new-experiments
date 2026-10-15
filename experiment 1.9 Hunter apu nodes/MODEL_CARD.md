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

## Initial Single-Node Results

The initial run completed 66 updates and 21,120 scheduled samples. Its best and latest
saved checkpoints both correspond to step 66. Monitoring Spearman improved
from 0.06407690096409865 before training to 0.08075104483874465 at step 66.
The parent scored 0.08394159463901718 on the same monitoring subset.
These are 1,616-basket monitoring results, not held-out or full-test results.
The last training loss was 0.6909430596977473.

The scheduler recorded a termination request from the owning account near
15:00 CEST on 2026-10-15, before the scheduled 16:11 CEST walltime endpoint.
Training logged a termination signal after completing and verifying the
step-66 checkpoint. This was not a full two-hour training run.

## Two-Node Continuation

Job `286615.hunter-pbs01` continued from step 66 on eight MI300As across two
Hunter nodes, starting at 15:21:45 CEST on 2026-10-15 with scheduled walltime
ending at 20:21:45 CEST. FP32 model weights, Adam and sampler state were
preserved; RNG streams were reseeded for eight workers. This is not a
bitwise-identical continuation. The global batch and precision contract
remain unchanged. The published snapshot section records the saved steps
included in the current upload.

The same allocation restarted at step 1203 with all eight rank RNG states,
FP32 model/Adam and sampler restored. It then saved latest checkpoints during
five-minute validation cycles, with a separate pre-expiry save.

## Final Continuation Results

PBS finished the allocation after exceeding its five-hour walltime, with
exit status -29 and recorded completion at 20:28:41 CEST on 2026-10-15.
The last recorded update was step 2183 / 698,560 scheduled samples. The final
verified recovery checkpoint is step 2172 / 695,040 samples, saved at
20:21:36 CEST. The final 11 updates are not recoverable from that checkpoint.
Best remains step 328 / 104,960 samples, monitoring Spearman
0.08534306347402562. These monitoring results are not held-out test scores.

## Contents

- `best/` and `latest/`: BF16 inference matrices, FP32 vectors, configuration
  and saved-step metadata for the published snapshot.
- `runs/`: original TensorBoard event files, visible in the hosted Metrics tab.
- `reports/`: aggregate training/validation history, hardware preflight,
  checkpoint verification and scheduler status.
- `source/`: original deployment/runtime archives and job-specific training sources.

Full FP32/Adam/RNG/sampler recovery checkpoints and private data are not public
uploads. Full recovery is handled separately through a private backup; its
verification receipt is not interchangeable with this public upload. These inference
exports alone cannot exactly resume training. No bitwise-resume certification
is claimed for Experiment 1.9.

Source: [Experiment 1.9 on GitHub](https://github.com/YLiu95/MSc-new-experiments/tree/main/experiment%201.9%20Hunter%20apu%20nodes).
Metrics: [Hosted TensorBoard](https://huggingface.co/YL95/experiment-1.9-hunter-apu-nodes/tensorboard).

This is research software for cross-sectional ranking, not a general-purpose
language model or a validated trading system. It does not establish future
investment performance and is not financial advice.