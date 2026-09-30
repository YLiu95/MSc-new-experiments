# Experiment 1.7

Task-conditioned whole-basket ranking, built from the corrected 1.6 data contract and 1.5 tensor-parallel primitives. The instantiated model has 10,193,843,857 parameters: width 5120, 80 heads, feedforward 20480, 20 temporal blocks and 12 cross-ticker blocks. TP=8 per node and DP across 2-8 nodes. Each GPU owns a model shard, not a complete 10B model replica.

## Frozen decisions

- Preserve corrected source-imputation masking, chronological splits, historical-only basket eligibility, masked forward labels, normalization, and SQLite nonrepetition sampling.
- Enable all 638,820 tasks, including the three previously held-out configurations. Chronological validation remains separate. No held-out-task generalization claim is permitted.
- Pairwise logistic ranking loss; primary metric is within-basket Spearman, equally aggregated by date, market and task.
- Use the frozen 1.6 primary monitoring subset for selection. This is explicitly not full validation or test evaluation. The larger suite is retained in the data source but not used for selection in this run.
- Global batch 320, microbatch 1, inherited learning-rate schedule, FP32 master weights and AdamW state, BF16 matrix operations, activation checkpointing, dropout 0.15 and identity dropout 0.10. No checkpoint warm-start.

## Run

Install `requirements.txt` in a scratch-backed virtualenv at `/net/tscratch/people/<user>/experiments/experiment_1.7_venv` (or set `EXPERIMENT_PYTHON_ENV` to another environment with the pinned dependencies). From this directory, `source env.sh`, then `python -m pytest tests -q`, then `python -m ranking17.launch submit --nodes 3` for a three-node allocation. Without `--nodes`, the request remains 2-8 nodes. The submitter rechecks the active reservation, requests at most ten hours, and caps the request before reservation expiry. Each batch allocation freezes its actual DP layout and code digest before execution.

The detached job publishes an allowlisted source backup, runs the full >10B worst-case memory and bitwise resume-next-update gate, then starts fresh training only after that gate passes. Gate failure terminates the allocation. Persistent per-rank logs and reports are under the private artifact root, grouped by Slurm job ID.

Training uses the earlier of actual job EndTime and reservation EndTime, reserving at least 20 minutes for final checkpointing and verified private backup. Slurm warning signals create a shared STOP file. Checkpoints require a digest-bound COMPLETE marker; recovery includes model, optimizer, step, per-rank RNG and sampler ledger. Explicit resumes require the same frozen configuration and parallel layout and append a restart record. Validation runs on step 1 and every five steps thereafter when there is sufficient time before shutdown. Each improved best checkpoint replaces the `best/` inference files on Hugging Face immediately after saving, with remote size and digest verification; no optimizer, sampler, or private data is included. The final publisher still handles the latest export and aggregate reports after the private backup.

Private backups: `/net/tscratch/people/tutorial042/experiments/experiment_1.7_backups/job-JOBID`, mode 700. Home has a 10 GiB hard quota and cannot hold the approximately 115 GiB recovery checkpoint. This verified second copy is on the same shared filesystem, not an independent disaster-recovery backup; it remains subject to scratch retention policy. Public source and aggregate reports: `YLiu95/MSc-new-experiments/experiment_1.7_asc_gpu_nodes`. Public inference: `YL95/experiment-1.7-asc-gpu-nodes`. If best=latest, publish best only. Raw panels, sample labels/predictions, sampler ledgers and optimizer recovery files never enter public uploads.

TensorBoard must run on the forwarded Jupyter/VS Code host using `tensorboard --host 127.0.0.1 --port 16006 --logdir /net/tscratch/people/tutorial042/experiments/experiment_1.7`. Forward port 16006 privately in VS Code. It needs restarting after that host's allocation ends.

The `ascgpu` and `ranker` packages are vendored provenance dependencies. Their old training/launch/publish entrypoints are not Experiment 1.7 commands. Use only `ranking17` entrypoints.