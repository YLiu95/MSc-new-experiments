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
  have not been independently certified. Twelve markets have no weekend
  sessions; IN has three market-wide weekend dates (2019-10-27: 3,316 quotes;
  2020-11-14: 3,401; 2025-02-01: 4,254), plausibly special sessions rather
  than stray timestamps. Invalid prices were recorded and masked, not
  forward-filled.
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
  two hours. Rechecked at 19:03:08: requests for 1, 2, 3, 4, 6 and 8 nodes
  at two, eight and twelve hours all forecast immediate starts; extra nodes
  rely on shared FLEX capacity and forecasts are not allocations. The user
  then approved an eight-node, up-to-twelve-hour request with a 60-second
  maximum wait and **manual stop/review after the screen**. The productive
  2/3/3-node layout, budget amendment and release policy are in
  [CAPACITY_AMENDMENT.md](CAPACITY_AMENDMENT.md). No overnight training job ID
  exists until the revised multi-node checks and actual `sbatch` succeed.
- Private TensorBoard 2.21 runs on 127.0.0.1:16007 on this Jupyter host;
  HTTP 200 and the saved smoke run were verified. No public tunnel was opened.
- Cross-node diagnostic `3211008` used all eight A100 nodes for three
  independent A/B/C groups (DP=16/24/24); Slurm reported COMPLETED/0 in
  58 seconds. BF16 score errors were zero and the worst reported gradient
  absolute error was 0.002686, within the 0.02 tolerance. All resources
  were released. This was a gradient/padding diagnostic, not training.
- First screen submission `3211032` started on eight nodes at 19:23:58 but
  exited FAILED/2 immediately with **zero optimizer updates**: `sbatch
  --export=NONE` removed `LD_LIBRARY_PATH` and the batch interpreter could
  not load `libpython3.11.so.1.0`. The error is preserved in the job's Slurm
  stderr. `env.sh` now restores the pinned Python, OpenSSL, SQLite, libffi
  and zlib library directories without reintroducing inherited CUDA/NCCL
  paths. A clean-environment rehearsal passed, and short detached preflight
  job `3211034` completed 0:0 in three seconds with all required runtime
  imports and empty stderr. The failed job ID remains in the submissions
  ledger; a replacement eight-node screen requires a new verified source
  commit and a new Slurm job ID.
- Second screen submission `3211041` started on all eight nodes at 19:28:28
  but exited FAILED/2 with **zero optimizer updates**: the clean batch's
  `srun` step could not resolve a bare `bash` executable. The launch now
  uses `/usr/bin/bash` and sets `SLURM_EXPORT_ENV=ALL` only after filtering
  the batch environment. Nested one-node `sbatch`/`srun` preflight `3211051`
  completed 0:0 in two seconds with empty stderr and the required imports.
  More importantly, clean eight-node `sbatch`/`srun`/three-group diagnostic
  `3211058` completed 0:0 in twelve seconds, producing A DP=16 and B/C
  DP=24 BF16 gradient checks within tolerance. Its IPv6 socket fallback
  warnings did not stop NCCL. Both failed screen IDs are retained in the
  ledger; neither trained a sample nor wrote a checkpoint.
- The inherited home quota is 10 GiB. A failed installation was moved to
  private scratch, with a home-path symlink; the separate pinned environment
  then installed and passed `pip check`. The 1.5 environment was left alone.
- A `pip-audit` scan of the full resolved lock initially found advisories in
  `pip`, `setuptools`, `pyarrow`, `pytest`, and `python-dotenv`. The direct pins
  were amended to published fixed versions, the complete CPU suite passed,
  and a second scan reported **no known vulnerabilities**. The scanner ran
  in a separate scratch-backed environment. TensorBoard 2.19 failed to start
  with the security-fixed setuptools because it imported `pkg_resources`;
  the pinned runtime was amended to 2.21, which has no such import. The
  refreshed complete lock was rescanned: **no known vulnerabilities**.
- GPU correctness check `3210831` (two A100s) matched a single-device BF16
  reference at both ranks with unequal informative counts and zero-weight
  padding; maximum measured score/gradient absolute error was 0 in this
  small diagnostic. The allocation was released.
- Worst-case synthetic profile `3210833` (eight A100s) ran 10 warmups and 50
  measured effective 320-sample updates at L=512/H=90/K=128. Instantiated
  parameters: 11,021,201. Mean measured step 2.94 seconds; checkpoint staging
  and verified BF16 export took 6.31 seconds. All eight ranks reported peak
  reserved memory 0.96094 GiB against the 34 GiB threshold, finite gradients
  and nonzero score variance. This is synthetic data, not ranking evidence.
- The first full-manifest real-data smoke `3210843` completed step 1 and wrote
  complete step-0/step-1 checkpoints in 7m13s. A separate eight-A100 resume
  reached step 21 and 6,720 scheduled attempts, all 320 informative at the
  logged step 20 (loss 0.801416, gradient norm 0.983774, score variance
  0.650020, 3.40 seconds per update). Best remained step 0 on the short
  validation comparison; this is not skill evidence. A second private copy of
  real best/latest/log artifacts was hash-verified: 41 files / 199,014,994 bytes.
- The loader completed with four workers per GPU but warned that each rank
  inherited one allowed CPU from Slurm's default task binding. Short job
  `3210927` confirmed `srun --cpu-bind=none` exposes exactly the 32 allocated
  CPUs. The detached batch script now uses this binding; its performance is
  not yet measured in a production job.

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
2. Confirm bound-CPU throughput during a production-shaped screen and test
  a full-model resume after the revised binding. Complete-manifest validation
  and inference export passed the real GPU smoke, but the screen must show
  declining loss before scaling.
3. Pass a real cross-node NCCL gradient/rendezvous check for the amended
  2/3/3 layout and recheck the eight-node scheduler start immediately before
  detached submission; abort within the chosen 60-second wait if not granted.
4. Review matched A/B/C learning screens and complete private backup evidence
   before approving a full 30,000-update comparison or confirmatory seeds.
5. After a selected trained model and rights review, verify the private source
   commit and publish exactly one best/latest pair and approved aggregate logs
   to the requested public HF repo. Complete frozen probes/test/bootstrap after
   selection, not as a basis for choosing the checkpoint.

This report should be extended with official Slurm job IDs, GPU-hours, durations,
peak memory per GPU, recovery tolerances, validation and remote digest evidence
as execution progresses. Smoke checkpoints exist privately; no official
training run or public model has been claimed yet.