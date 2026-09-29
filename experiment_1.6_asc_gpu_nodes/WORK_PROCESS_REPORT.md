# Experiment 1.6 Work Process

Date: 2026-09-29 (UTC+02:00). Status: implementation and CPU validation;
overnight GPU training and final publication require an explicit node-count and
queue-wait decision. No scheduler forecast below reserves a node.

## Evidence obtained

- The requested private GitHub repository `YLiu95/MSc-new-experiments` exists,
  is private, uses `main`, and grants the authenticated account push access.
  The requested public HF model repo did not yet exist when checked. The HF
  token identifies `YL95`; the pinned training dataset is private and declares
  ODC-BY in its card. No destination or license is treated as proof of vendor
  price-data redistribution rights.
- The pinned 1.5 private source cache reproduced a FP64 training-only scale of
  3.4060786363283126 on 76,479,949 finite returns. New historical-streak
  indexes include 13 markets and 24,022 training-seen identities. The calendar
  union has no gaps of ten or more weekdays, but exchange-session calendars
  have not been independently certified. Invalid prices were recorded and
  masked, not forward-filled.
- All 638,817 training tasks are historically feasible; issuing 320 scheduled
  samples (including catalogue setup) took 2.611 seconds on the CPU tunnel
  host. This is a sampler diagnostic, not a full data-loader throughput or
  GPU-memory result.
- Frozen validation: 51,712 primary baskets (1,616 monitoring) and 19,456
  held-out-combination baskets. Primary has 3,352,558 observed and 1,554
  missing label occurrences, 99 degenerate basket correlations, and 3 missing
  market/task cells. The baseline primary macro Spearman is -0.0001899445 for
  positive momentum, +0.0001899445 for reversal, 0 for constant scores.
- The current CPU-only JupyterHub job is scheduled to end at 2026-09-29
  23:31:41; the separate `training` GPU reservation ends on 2026-09-30 at
  08:00. The batch job must finish private backups before 07:40 or earlier
  if its own walltime expires sooner. A Jupyter restart does not extend either.
- Slurm `sbatch --test-only` at 18:25:51 predicted immediate starts for 1, 2,
  and 3 nodes, each with eight A100s, 32 CPUs and 256 GiB RAM per node for
  two hours. Forecasts are workload-dependent. No overnight training job ID
  exists until the user selects nodes and maximum wait and the actual `sbatch`
  submission succeeds.
- The inherited home quota is 10 GiB. A failed installation was moved to
  private scratch, with a home-path symlink; the separate pinned environment
  then installed and passed `pip check`. The 1.5 environment was left alone.
- A `pip-audit` scan of the full resolved lock initially found advisories in
  `pip`, `setuptools`, `pyarrow`, `pytest`, and `python-dotenv`. The direct pins
  were amended to published fixed versions, the complete CPU suite passed,
  and a second scan reported **no known vulnerabilities**. The scanner ran
  in a separate scratch-backed environment.

## Correctness already checked

Focused CPU tests cover the exact 638,820 task-space count, held-out removal,
signed labels and H=90 split endpoints, future-independent basket eligibility,
arbitrary-precision basket ranks and deterministic ledger restore, ticker
equivariance and temporal padding, informative-pair normalization, UNK and task
gradients, tied prediction metrics, manifest freezing, best/latest export
round-trip and tamper detection, private backup allowlists, and a CPU two-rank
gradient comparison with unequal valid samples and padding. A real-format
tiny-panel run restored from a checkpoint and matched its next update against
an uninterrupted run. This is **not** a full-model GPU resume proof.

## Open gates

1. Confirm each exchange calendar and public derivative/identity rights.
2. Run BF16 DDP/memory/50-update GPU profiles (all eight devices), benchmark
   four workers per GPU, real-data continuation, and complete-manifest export.
3. Review updated Slurm forecasts; obtain the user's node count, maximum
   allocation wait, and stage budget before detached training submission.
4. Review matched A/B/C learning screens and complete private backup evidence
   before approving a full 30,000-update comparison or confirmatory seeds.
5. After a selected trained model and rights review, verify the private source
   commit and publish exactly one best/latest pair and approved aggregate logs
   to the requested public HF repo. Complete frozen probes/test/bootstrap after
   selection, not as a basis for choosing the checkpoint.

This report should be extended with actual Slurm job IDs, GPU-hours, durations,
peak memory per GPU, recovery tolerances, validation and remote digest evidence
as execution progresses. No checkpoint or public model has been claimed yet.