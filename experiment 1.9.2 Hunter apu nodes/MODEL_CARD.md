---
tags:
- tensorboard
- finance
- time-series
---
# Experiment 1.9.2

Two continuations of Experiment 1.9 step 328, each using six Hunter MI300A nodes
and 24 GPU workers. Both hide stock identity throughout training. Arm A samples
feasible markets uniformly; arm B weights feasible markets by included-stock
training-period stock-days.

The global batch is 1,536 per arm, with a 1,000-update budget and cosine learning
rate 6e-5 to 6e-6. Adam and parent basket-consumption records are preserved.
Selection uses the last five hidden-ID validation scores; the final test stays
sealed until both arms finish and checkpoint selection is frozen.

## TensorBoard

Select arm-A and arm-B together for like-for-like curves. Steps count new updates
since parent step 328. reference-1.9.1-ids-on is an explicitly labeled historical
reference, not another 1.9.2 arm. Original historical event files are also preserved.

Original event files and approved training logs are synchronized immediately,
then every 300 seconds, with a final upload. The local Windows relay depends on
the PC, VPN and authenticated SSH tunnel remaining available. Repository contents
are point-in-time snapshots, not proof that training has completed.

## Limitations

The hash-selected 10% stock holdout is excluded only from new 1.9.2 training;
the parent was exposed to the full universe. Training-era included and held-out
comparisons use 1,248 matched feasible baskets, while validation retains all
1,616 monitoring baskets. The data are survivors-only and do not establish
point-in-time exchange membership or unbiased delisting returns. Differences
below 0.004 are inconclusive; per-market claims require paired time-block analysis.

Public artifacts exclude optimizer state, RNG, sampler databases, private panels,
stock holdout lists and per-basket cohort definitions. Full recovery remains private.