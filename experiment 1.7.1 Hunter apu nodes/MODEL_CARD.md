---
library_name: pytorch
tags:
- tensorboard
- time-series
- ranking
- rocm
---
# Experiment 1.7.1: Hunter MI300A

Task-conditioned whole-basket ranker, 10,193,843,857 parameters. TP=4;
inference requires all four weight shards and the published model configuration.
FP32 training weights/Adam state, BF16 compute, original dropout semantics,
global batch 320 and microbatch 1.

## Final Observed Results (2026-10-15)

Six-node job `286067.hunter-pbs01` passed its 24-worker recovery preflight and
restored the step-192 parent with Adam and sampler progress preserved. It
completed 218 updates, steps 193 through 410, then failed during checkpointing
at 07:13:35 CEST with PBS exit status 127. There is no clean-completion record.

| Published export | Step | Monitoring Spearman | Scheduled samples |
| --- | --- | --- | --- |
| `best/` | 305 | 0.08394159463901718 | 48,640 |
| `latest/` | 355 | 0.06689862950127207 | 64,640 |

These scores use the monitoring subset, not full validation or a held-out test.
The final logged update was step 410 (loss 0.6902306942269206), but its weights
were not saved: 55 updates and 17,600 scheduled samples after step 355 are not
recoverable from the retained checkpoint. No step-410 weights are published.

Full FP32/Adam/RNG/sampler recovery at step 355 and its verified second copy
remain on Hunter's temporary workspace, not in this public repository. The
published exports are inference-only and cannot provide exact training resume.
The step-410 error reports a missing checkpoint staging directory; rank-local
checkpoint timing is a suspected cause, not a confirmed diagnosis or a fix.

## Provenance and Limitations

Initialization uses Experiment 1.7 latest step 153 at source revision
`ecf683c5cfd3f49c9fbf2ce5e53f2f637dfce9b9`. Source matrices were BF16 inference
exports. Adam, RNG and sampler reset were explicitly approved; this is not an
exact continuation, and earlier samples may be reused. Hunter recovery after
the warm start retains optimizer, sampler, RNG and training state privately.

Checkpoints with a `migration` configuration record use the subsequently
approved two-to-six-node handoff. That handoff preserves FP32 weights, Adam,
sampler progress, step count and global batch 320; it does not repeat the
step-153 initialization. RNG streams are reseeded per DP replica, staying
aligned within each TP4 group to preserve dropout behavior. This topology
change is not bitwise-identical continuation. A full 24-worker recovery gate
is required before migrated training proceeds.

All 638,820 task configurations are trainable. Selection uses the frozen 1.6
primary monitoring subset with chronological validation, not held-out-task
generalization or full validation/test. Training data are survivor-biased;
do not interpret monitoring scores as investment performance or financial advice.

Pinned data revision: `bcbbefdbe2313673895eb1a0d354747a9f1624fa`.
The source dataset declares ODC-BY and Yahoo provenance; the data owner has
disputed that attribution. Assess data rights independently before redistribution
or commercial use. Exact inference also needs the private identity vocabulary
and recorded normalization metadata.
The original corrected source-imputation and future-label availability rules
are preserved. Label availability is never used as an attention mask.

Only inference weights, configuration, aggregate reports and TensorBoard events
are published here. See the Metrics tab for hosted TensorBoard. Consult
`reports/` for actual completed preflight/training evidence; code readiness
alone is not a successful hardware test or training result.