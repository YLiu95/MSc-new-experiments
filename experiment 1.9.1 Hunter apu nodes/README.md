# Experiment 1.9.1: Twelve-Node Hunter Continuation

Starts from Experiment 1.9 best step 328, 104,960 scheduled samples,
monitoring Spearman 0.08534306347402562. The private recovery revision is
`7a17855a414de2241469c1e3fd78f5bf351a69dc`; manifest SHA-256 is
`133e851f4e4de23886b173f4c2874c8a9645e2531a185edde618256f607f2b01`.
Full FP32 weights, Adam moments and sampler progress are preserved. RNG is
reseeded for 48 workers, so this is not bitwise continuation.

## Training Contract

- Existing interactive PBS job `286972.hunter-pbs01`, twelve MI300A nodes,
  seven-hour walltime. No replacement allocation is submitted.
- Four GPU workers per node, TP1/DP48, 24 physical CPU cores per worker.
- Global batch 1536, microbatch 2, 16 accumulated microsteps per GPU.
  This changes the parent's global batch of 320; the learning rate is not scaled up.
- Same-shaped sampler groups are cost-balanced across workers without padding
  or duplicated samples. Four CPU input threads per worker use pinned buffers.
- Per-update LR is `min(original_schedule(step), 0.00006)`.
- BF16 compute, FP32 master weights/Adam, activation checkpointing, fused AdamW,
  128 MiB DDP buckets and gradient accumulation before communication.
- Every worker must pass a worst-case microbatch backward/Adam and memory gate.
  The synthetic update is rolled back before restoring the parent recovery.

The detached Hunter watcher checks the existing queue every five seconds and
launches once through PBS-authorized attachment when nodes are assigned.
**Keep the original `qsub -I` terminal open.** Closing that session can release
the allocation. The watcher is independent of this chat, not of PBS ownership.

The deadline is derived from actual PBS start time. Final saving starts with
a conservative measured I/O budget, targeting worker completion at least
120 seconds before allocation expiry. Routine latest saves follow five-minute
validation cycles. Best/latest full recovery and inference exports are verified
before unprotected generations are retired. Hard failures can lose unsaved updates.

## Monitoring And Backups

Live scheduler/startup/training stream:
`experiment191/artifacts/job-286972.hunter-pbs01/control/watch.log` on Hunter.
Per-rank logs and TensorBoard events are retained under the same job root.

Source: [GitHub](https://github.com/YLiu95/MSc-new-experiments/tree/main/experiment%201.9.1%20Hunter%20apu%20nodes).
Public model and metrics: [Hugging Face](https://huggingface.co/YL95/experiment-1.9.1-hunter-apu-nodes).
[Hosted TensorBoard](https://huggingface.co/YL95/experiment-1.9.1-hunter-apu-nodes/tensorboard).
Full recovery destination: private `YL95/experiment-1.9.1-hunter-recovery`.

The local relay uploads metrics approximately every minute when not transferring
weights, public checkpoints approximately every ten minutes and at completion,
then full best/latest recovery privately. Keep the PC, VPN, SSH tunnel and relay
running for cloud synchronization. Training and its queue watcher run on Hunter.
Cloud receipts are written only after remote file hashes and privacy are verified.
Temporary publication snapshots are released between completed relay attempts.

The source archive bundles unchanged runtime dependencies and the prior trainer's
checkpoint helper. In this development workspace, tests need this directory,
`experiment 1.9`, and the adapted Experiment 1.7.1 runtime on `PYTHONPATH`.
The deployed/GitHub source is self-contained apart from the installed packages.

Local and Hunter policy/recovery/launcher tests passed before arming the watcher.
Actual 48-GPU utilization, throughput, memory headroom and hosted training curves
require a running allocation; queued status is not evidence that those gates passed.