# Experiment 1.6: Eight-Node Screen Amendment

Recorded 2026-09-29, before official training submission. The user chose an
eight-node, up-to-twelve-hour request, prioritizing the node count and then
available hours, with a **60-second maximum allocation wait**. The user also
explicitly chose to **stop after the learning screen for manual review**.
These selections amend the plan's initial one-node layout and proposed screen
GPU-hour ceiling. They do not authorize additional training seeds, a larger
model, a changed ranking objective, or automatic main-stage continuation.

| Arm | Nodes | A100 GPUs | Layout | Stream |
| --- | ---: | ---: | --- | --- |
| A | 2 | 16 | TP=1, DP=16 | Fixed (256,7,64) |
| B | 3 | 24 | TP=1, DP=24 | Variable seed 1337, no task conditioner |
| C | 3 | 24 | TP=1, DP=24 | Identical scheduled variable stream to B, task-conditioned |

The eight-node batch job runs three independent cross-node PyTorch process
groups with separate rendezvous addresses/ports, sampler ledgers, runs and
checkpoint roots. All requested nodes have work. B and C have the same DP
layout and seeded task stream. The registered 320 **real** attempts per update
and four-sample task visits remain unchanged. DP=16 needs five four-sample
microsteps. DP=24 needs four; the final physical step has 64 zero-weight
execution-padding samples that create no sample IDs and contribute neither
loss nor gradient. Global informative-sample sum/count normalization is
unchanged. Continuation requires the same per-arm DP layout.

The job wall request is at most 12 hours **only when that still fits within the
current verified reservation**, a 60-second queue wait, and a five-minute
scheduling margin. Eight nodes × eight GPUs × twelve hours would be a **768
GPU-hour upper allocation ceiling**, not a forecast of actual GPU use or a
justification to occupy GPUs while idle. The screen stops at 2,000 completed
updates per arm (earlier at its conservative deadline) and exits/relinquishes
the entire allocation as soon as all arms and private backups finish. It does
not occupy unused walltime waiting for manual main-stage approval.

Before submission, the eight-node layout needs a real short multi-node
communication/gradient check and a current `sbatch --test-only` prediction.
Use `sbatch` without `--wait`, persist the numeric job ID, and query it after a
new JupyterHub connection. The live overnight `training` reservation ends
2026-09-30 08:00 UTC+02:00; configured two-day partition maximum and FLEX
availability do not promise access later. Compute the training stop using the
earlier of actual batch EndTime and the reservation end, reserving at least
15% of job walltime and at least 40 minutes; verify private best/latest/log
backups at least 20 minutes before that earlier shutdown. Checkpoint pointers
advance only after full integrity checks.

The main comparison remains **unapproved pending the user's screen review**.
If approved later, its hours, Slurm request, software config, seed policy and
common completed prefix require a separate dated decision. A request for a
12-hour walltime must not be described as 12 hours of completed training.