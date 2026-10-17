---
tags:
- tensorboard
- finance
- time-series
---
# Experiment 1.9.2

Two continuations of Experiment 1.9 step 328, each using three Hunter MI300A nodes
and 12 GPU workers in the six-node restart. Both hide stock identity throughout training. Arm A samples
feasible markets uniformly; arm B weights feasible markets by included-stock
training-period stock-days.

Job 287990 failed at update 300 due to shared disk quota; both last complete recoveries are update 250.
The six-node job 287999 resumes those recoveries, not the unsaved update-300 state.
Optimizer, sampler, selection state and cumulative update numbers are preserved;
the 32-to-12 worker topology change requires RNG reseeding and is not bitwise continuation.

The global batch is 1,536 per arm, with a 1,000-update budget and cosine learning
rate 6e-5 to 6e-6. Adam and parent basket-consumption records are preserved.
Selection uses the last five hidden-ID validation scores; the final test stays
seeded initially with the baseline score. The final test stays sealed until both arms reach
the update budget or early stopping; quota, signal and deadline stops do not qualify.

## TensorBoard

Select arm-A and arm-B together for like-for-like curves. Steps count new updates
since parent step 328. reference-1.9.1-ids-on is an explicitly labeled historical
reference, not another 1.9.2 arm. Original historical event files are also preserved.

Original event files and approved training logs are synchronized immediately,
then every 300 seconds, with a final upload. The local Windows relay depends on
the PC, VPN and authenticated SSH tunnel remaining available. Repository contents
are point-in-time snapshots, not proof that training has completed.
Complete stopped-run recoveries are uploaded separately to the private recovery repository.
Cleanup requires verified cloud receipts and retains the best parent and current best/latest
arm recoveries locally. Quota admission reserves concurrent save space; cloud outages do not
authorize deletion and cannot be assumed to survive loss of relay dependencies.

## Limitations

The hash-selected 10% stock holdout is excluded only from new 1.9.2 training;
the parent was exposed to the full universe. Training-era included and held-out
comparisons use 1,248 matched feasible baskets, while validation retains all
1,616 monitoring baskets; 23 of 101 market/task cells lacked enough matched dates.
Weighted market choice consumes RNG differently, so training baskets are not paired between arms.
Common evaluation baskets remain paired. From the update-250 restart, unused known stock-ID
rows are held at their resumed values despite inherited Adam momentum; other model updates
can still change the IDs-on reference metric. Historical audit ledgers deleted before this run
are explicitly unavailable, not estimated. The data are survivors-only and do not establish
point-in-time exchange membership or unbiased delisting returns. Differences
below 0.004 are inconclusive; per-market claims require paired time-block analysis.

Public artifacts exclude optimizer state, RNG, sampler databases, private panels,
stock holdout lists and per-basket cohort definitions. Full recovery remains private.