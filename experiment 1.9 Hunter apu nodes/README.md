# Experiment 1.9: Depth-Pruned Hunter Ranker

Parent: Experiment 1.7.1 best step 305, monitoring Spearman 0.08394159463901718,
Hugging Face revision `586d6a4eba7f233b1070fc59acfef8857aad6444`.
The selected model has **1,069,302,417 parameters**: temporal blocks 0 and 19,
cross-sectional block 11, and all original embeddings, conditioning and head.
Width 5120, 80 attention heads and feedforward width 20480 are unchanged.
The first/last temporal blocks span the source stack, and the final cross block
is retained next to the trained head. This is a deterministic structural choice,
not a validation-selected or importance-ranked optimum. Pruning changes the
function; the original 10B score is not the child's score.

Extraction verifies the parent manifest and payloads, preserves Q/K/V and FiLM
component ordering while merging TP4 shards into TP1, and records every tensor's
source layer. BF16 inference matrices initialize FP32 trainable parameters.
Adam, RNG, sampler and learning-rate warmup restart at zero; earlier training
samples may recur. The original pinned panel and monitoring split are unchanged.

One exclusively allocated Hunter node runs four DP replicas, one per MI300A,
with 24 physical CPU cores per worker, BF16 compute, FP32 Adam/master weights,
activation checkpointing, global batch 320, microbatch 1, and original dropout.
The full-size worst-case backward/Adam smoke test runs on all four GPUs, then
its update is rolled back before initial monitoring validation and training.
No bitwise resume certification is claimed by that smoke test.

Rank zero broadcasts checkpoint/validation decisions. Complete best and latest
recovery checkpoints and inference exports are retained on Lustre. New files
are staged, hashed and atomically promoted before older unprotected generations
are retired. Checkpoints are attempted every two minutes between updates,
validation every five minutes, and a checkpoint near allocation expiry.
Training continues until PBS terminates it. Hard termination cannot guarantee
saving the in-flight update; `latest` means the last complete verified save.
No automatic restart, early training cutoff or replacement allocation is used.

The artifact root is
`/lustre/hpe/ws12/ws12.a/ws/vdbg1007-hunter-exp171/experiment19/artifacts/job-286464.hunter-pbs01`.
Its `best.json` and `latest.json` point to full recovery checkpoints under
`checkpoints/`; inference-only exports are under `exports/`. The root also
contains `job.log`, `history.jsonl`, `validation.jsonl`, and TensorBoard `runs/`.

Existing job `286464.hunter-pbs01`, node `x1000c0s1b0n0`, started at
2026-10-15 14:07:10 CEST; its recorded walltime is 02:03:56, with scheduled end 16:11:06 CEST.
Only its authorized job-bound SSH route is used. The source Experiment 1.7.1
artifacts and backups are not modified. Temporary Lustre storage is not an
independent off-site backup. Raw data, optimizer, RNG and sampler files must
not be included in public model uploads.

Runtime dependencies are the unchanged `ranker`, `ascgpu` and `ranking17`
packages from the verified Experiment 1.7.1 source. Hunter imports them from
`source-six-node-20261014` and reuses its installed environment. The GitHub
backup includes the exact runtime packages from the run's source archive.
In the original local workspace, tests require this directory and
`experiment 1.7.1/source/experiment 1.7.1 Hunter apu nodes` on `PYTHONPATH`;
the self-contained GitHub backup needs only its own directory.

Local tests: `python -m pytest tests/test_experiment19.py -q`.
Deployment entrypoint: `bash -l run.sh`, only within the named allocation.
The login shell initializes Hunter's module command before loading PyTorch.
The idle PBS wrapper was paused to prevent its obsolete 15:00 sleep deadline
from releasing the allocation before the unchanged PBS walltime expires.
RCCL uses `NCCL_NET=Socket` because the module's forced AWS Libfabric backend
was unavailable in the SSH-launched environment. GPU peer communication is
not disabled. A four-GPU all-reduce passed with this setting. Each worker
starts through `pbs_attach` so PBS tracks and terminates its session.
The corrected source tests passed on Hunter: 5 tests in 39.48 seconds.

## Verified Start

On 2026-10-15, the four-GPU full-model backward/Adam check passed in
9.20 seconds, with peak reserved memory 16.42 GiB per MI300A. The synthetic
update was rolled back. Initial child monitoring Spearman was
0.06407690096409865, measured before training, rather than inherited from
the parent. At 14:49:10 CEST all four GPUs showed 99-100 percent activity.
Real updates were confirmed; steps 18-20 achieved 53-59 samples/second.
These are startup observations, not final results or sustained-run averages.
The first trained full recovery checkpoint, step 20 / 6,400 scheduled samples,
was hash-verified alongside the protected step-zero baseline best checkpoint.
Periodic checkpoint and validation decisions ran until the termination request.

## Recorded Stop And Backups

Training stopped after 66 updates / 21,120 scheduled samples. Both best and
latest complete checkpoints are step 66, with monitoring Spearman
0.08075104483874465. The scheduler recorded termination requested from the
owning account on `hunter-login04`, near 15:00 CEST on 2026-10-15, before the
scheduled walltime endpoint. No automatic restart was performed.

Source destination: [Experiment 1.9 on GitHub](https://github.com/YLiu95/MSc-new-experiments/tree/main/experiment%201.9%20Hunter%20apu%20nodes).
Model destination: [Experiment 1.9 on Hugging Face](https://huggingface.co/YL95/experiment-1.9-hunter-apu-nodes).
Metrics: [Hosted TensorBoard](https://huggingface.co/YL95/experiment-1.9-hunter-apu-nodes/tensorboard).

`publish19.py` publishes allowlisted source files, best/latest inference
checkpoints, aggregate metrics, original TensorBoard events and provenance.
Full Adam/RNG/sampler recovery files and private panels stay on Hunter.
Local publication receipts record the verified GitHub/Hugging Face commits.