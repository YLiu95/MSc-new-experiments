# Experiment 1.6 Work Process

Date: 2026-09-29 (UTC+02:00). Status: first eight-node screen stopped and
invalidated after a source-imputation audit; corrected data prepared separately.
No model has been published. No scheduler forecast below reserves a node.

## Evidence obtained

- The requested private GitHub repository `YLiu95/MSc-new-experiments` exists,
  is private, uses `main`, and grants the authenticated account push access.
  The requested public HF model repo did not yet exist when checked. The HF
  token identifies `YL95`; the pinned training dataset is private and declares
  ODC-BY in its card. No destination or license is treated as proof of vendor
  price-data redistribution rights.
- The first, now **invalidated**, 1.5 return-cache preparation had a scale of
  3.4060786363283126 on 76,479,949 training returns and 24,022 identity
  rows. It unknowingly retained source-forward-filled prices, so these are
  not valid 1.6 scientific measurements. The calendar union has no gaps of
  ten or more weekdays, but exchange-session calendars are not independently
  certified. Twelve markets have no weekend sessions; IN has three market-wide
  weekend dates (2019-10-27: 3,316 quotes; 2020-11-14: 3,401;
  2025-02-01: 4,254), plausibly special sessions rather than stray timestamps.
- Corrected private root `experiment_1.6_masked` uses the pinned Parquet
  `flag_imputed` column: 36,089 flagged prices across all 13 markets (7,758
  in train, 9,149 in validation) and both touching returns are rejected.
  This excludes 61,505 previously finite returns. The new FP64 training-only
  scale is 3.40288677815176 on 76,467,613 finite returns; the training
  identity vocabulary has 24,010 rows. The dataset card explicitly states
  that its Yahoo-derived universe is **survivors only**.
- All 638,817 training tasks are historically feasible; issuing 320 scheduled
  samples (including catalogue setup) took 2.611 seconds on the CPU tunnel
  host. This is a sampler diagnostic, not a full data-loader throughput or
  GPU-memory result.
- New frozen validation: 51,712 primary baskets (1,616 monitoring) and
  19,456 held-out-combination baskets with a different manifest digest.
  Corrected primary has 3,344,940 observed and 9,172 missing label
  occurrences, 123 degenerate correlations and 3 missing market/task cells.
  Corrected baseline macro Spearman is momentum +0.001052323, reversal
  -0.001052323, constant scores 0; the initial baseline is invalidated.
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
  [CAPACITY_AMENDMENT.md](CAPACITY_AMENDMENT.md). Failed starts and the
  stopped diagnostic below have retained IDs but no valid model result.
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
- Job `3211065` ran on all eight nodes and reached A=275, B=220, C=220
  updates before the source-card imputation policy error was discovered.
  The stop marker caused an orderly exit at 19:36:26 (COMPLETED/0).
  Each arm's latest/best/logs were rehashed in a second private home copy:
  A 310,841,848 bytes, B 310,451,170 bytes, C 311,690,684 bytes.
  These old-root checkpoints and baselines are **invalidated diagnostics**;
  they are not eligible for resume, selection or HF publication. The old
  scratch root and NFS backups were preserved without overwriting.
- Corrected one-node/eight-A100 smoke `3211083` (new root and policy) completed
  0:0 in 7m19s. It ran the entire corrected primary suite at step 0 and
  after one real optimizer update, keeping trained best/latest at step 1.
  Macro Spearman moved from -0.009125774 to -0.009068159; this tiny,
  single-update difference is an operational check, **not skill evidence**.
  A second private copy of its best/latest/logs (39 files, 309,185,430
  bytes) was rehashed at 20:01:38, before its 20:13:38 backup deadline.
  TensorBoard port 16007 now serves only corrected event paths (HTTP 200).
- The source dataset card declares ODC-BY for its compilation but says the
  underlying Yahoo Finance data is intended for personal/noncommercial use.
  Rights to publish derived weights and the ticker vocabulary have not been
  independently confirmed, so public HF upload remains gated.
- The inherited home quota is 10 GiB. A failed installation was moved to
  private scratch, with a home-path symlink; the separate pinned environment
  then installed and passed `pip check`. The 1.5 environment was left alone.
  The existing pip cache was also moved without deletion to private scratch,
  keeping its home symlink and freeing roughly 1.7 GiB for corrected-run
  independent NFS backups.
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