# Experiment 1.9.2

Two identity-hidden continuations from the verified Experiment 1.9 step-328 full
recovery. The current request is six nodes for 24 hours, in a native PBS batch job.
Each arm receives three disjoint nodes and 12 GPU workers. Batch 1536 means 128
baskets per worker and 64 two-basket accumulation microsteps. Arm B alone changes
market probabilities. `EXPERIMENT192_JOB` pins monitoring and publishing to the
submitted job; batch workers receive this identity from `PBS_JOBID`.

## Automatic Launch

PBS starts `run.sh` itself, without an interactive allocation shell, compute SSH,
or nested `pbs_attach`. Submission requests six exclusive nodes, 96 CPU cores per
node, four GPUs per node, and disables rerunning. The source is immutable and
hash-checked before startup; duplicate-start locks prevent a second copy.
The job ID, start time and 24-hour deadline come from the actual PBS job.
The detached controller observes progress but never launches a competing process.
A nonzero batch exit is recorded immediately and propagated through the notifier.

Training first runs its full-size backward/Adam gate on every worker, rolls that
update back, reloads the parent FP32/Adam state, and saves an initial recovery
checkpoint. The user authorized replacement allocations when necessary; retries
must first diagnose the error and verify the old job and its workers have stopped.

## Failed Allocation Evidence

Job `287925.hunter-pbs01` was interactive. PBS recorded two execution attempts,
final start 2026-10-17 03:45:19 CEST, end 03:46:43, exit status 1, and 37 seconds of
accounted wall time for the final attempt. Neither arm passed preflight or updated.
Arm-B workers reported `pbs_attach: tm_attach: no matching job found (17006)`.
The old launcher attempted to attach workers across an allocation that restarted;
the evidence does not establish why PBS ended or restarted that interactive job.
The reported home-directory output files and login-node tracejob records were not
available. The replacement removes the interactive dependency and nested attach.
This fixes the observed launch path, but GPU execution still requires validation.

Final failed-job logs were verified at Hugging Face commit
`6a35f5424ccf6d42a9179a1da6042c76ace8b053`. Old artifacts are retained unchanged.

## Evaluation

Validation uses the unchanged 1616 baskets. Included-vs-held-out training-era
comparisons use 1248 jointly feasible baskets per cohort. The SHA256 stock split
excludes 10% of symbols from both arms' new training. Parent exposure remains a
limitation. Filtering preserves the parent task catalogue, checks candidate
combinations against the original ledger, and keeps separate new counters.

Validation runs every 50 updates and matched stock evaluation every 200. The
initial evaluation is also recorded. Selection averages five hidden-ID scores;
early stopping counts six validations without improvement exceeding 0.002 after
the five-score window is available. Best-smoothed and latest full recoveries are
retained. Final-test access is guarded until both arms freeze selection, with a
one-hour remaining-time gate. Final evaluation is never used for selection.

## Monitoring

The independent Windows log relay uploads immediately and every 300 seconds,
retaining original event files and verified cloud commit receipts. Public logs
exclude stock lists, cohort definitions, per-basket rows, optimizer, RNG and sampler.
The PC, VPN and SSH tunnel must remain available for these cloud uploads.

TensorBoard uses the local verified event mirror at http://127.0.0.1:16006.
Because the server is on this Windows PC, no separate VS Code port forwarding is
needed for that URL. Remote log transfer uses the authenticated SSH forwarding
endpoint. Both arm runs use the same tags and new-update step axis; the historical
IDs-on reference is explicitly labeled.

## Validation Status

The six-node revision passes 25 local tests. Six additional native Bash tests run
on Hunter and exercise every arm/node-rank branch and post-module Socket transport.
GPU hardware gates and actual first updates must be checked after allocation;
unit tests and scheduler running state do not establish successful training.

## Verified Setup

The old immutable source `experiment192/source-287925-transport` is retained for
diagnosis. The replacement uses a separately validated, immutable batch bundle.

The first public log snapshot was hash-verified at Hugging Face commit
`cd0395e89abe2b2dab40d049f29a3618f33824d2`; the persistent relay then verified
`8e2377d61e8d6dfdbdf88c1f1bd0965ef039d4f0`. Local receipts are in
`publication/logs/metrics.json`, with relay output and backup status alongside.

Both the hosted TensorBoard and local browser at http://127.0.0.1:16006 displayed
arm-A, arm-B and the aligned historical reference. The new arms currently contain
configuration scalars only, not training curves. Hosted original historical events
also appear as `reference-original`; deselect that run for the aligned comparison.

Hosted view: https://huggingface.co/YL95/experiment-1.9.2-hunter-apu-nodes/tensorboard

The VS Code task `Experiment 1.9.2 TensorBoard :16006` serves the local mirror.
No VS Code Ports forwarding entry was created because the service is already local.
The global experiment instructions now require this monitoring setup for future runs.

## Allocation-Triggered Copilot

The earlier CLI dispatch implementation has been removed at the user's request.
`wake192.py watch --phase allocation|training|completion` only monitors the private
publication endpoint and exits on the selected event or a failure. It never invokes
Copilot CLI, submits a prompt, or creates another session. Training readiness requires
both preflights, completed initial checkpoint pointers and at least three updates.

Run these finite watchers through the current conversation's terminal tool. Their
completion/failure notifications return to that conversation, as the previous log
relay notification did. This is not a supported force-resume API for an idle chat;
delivery and agent continuation depend on VS Code. Native PBS startup does not depend
on the assistant waking. Keep this workspace, PC, VPN, tunnel and Copilot available.
No security or approval settings are changed, and arbitrary unattended repair is not
guaranteed. A later authorized retry must preserve healthy jobs and verify cleanup.