# Experiment 1.9.2

Two identity-hidden continuations from the verified Experiment 1.9 step-328 full
recovery. The six-node restart uses three disjoint nodes and 12 GPU workers per arm.
Batch 1536 means 128 baskets per worker and 64 two-basket accumulation microsteps. Arm B alone changes
market probabilities. `EXPERIMENT192_JOB` pins monitoring and publishing to the
submitted job; batch workers receive this identity from `PBS_JOBID`.

## Current Status and Checkpoint Cleanup

Job `287990.hunter-pbs01` failed at update 300 because the Lustre group quota
could not accommodate an atomic recovery save. Both latest complete recoveries
are step 578, or 250 new updates; update 300 is not a resumable checkpoint.
Training has not restarted. The separate six-node, 24-hour interactive request
`287999.hunter-pbs01` was still queued at the final cleanup check on 2026-10-17.

User-authorized cleanup removed 25 obsolete checkpoint, export and warm-start
directories, reclaiming an additional **272.7 GiB** after the earlier 133 GiB
duplicate removal. Group quota usage is **67,464,280 / 500,000,000 KiB**, leaving
**412.5 GiB free**. These are quota measurements, not filesystem-wide capacity.

The only remaining full checkpoint directories, relative to `HUNTER_WORKSPACE`, are:

```text
experiment19/artifacts/job-286615.hunter-pbs01/checkpoints/step_00000328_000000104960
experiment192/artifacts/job-287990.hunter-pbs01/arm-A/checkpoints/step_00000578_000000488960
experiment192/artifacts/job-287990.hunter-pbs01/arm-B/checkpoints/step_00000578_000000488960
experiment192/artifacts/job-287990.hunter-pbs01/arm-B/checkpoints/step_00000328_000000104960
```

These retain the best 1B parent, both arms' latest FP32 model/Adam/RNG/sampler
states, and Arm B's selected-best recovery required by resume and final selection.
Arm A's best equals its latest. All four passed complete payload hash verification
before and after deletion; sampler database checks passed and pointers are unchanged.
Matching exports, all datasets, source, logs, TensorBoard events and reports remain.
The final scan found no other files over 1 GiB outside this retained set and its exports.

The private audit records are `experiment192/cleanup-20261017/plan.json` and
`result.json` on Hunter, including removed checkpoints' metadata and verification
hashes. Obsolete payloads were deleted directly under the user's authorization;
no new Hugging Face checkpoint backup was made for this cleanup. Historical
pointers outside the retained set describe past runs, not available recovery targets.
Final failed-run logs were cloud-verified at commit
`b6d9cb085826444cbbd20807e027b85c4f588a35`; that log relay has finished.

## Sixteen-Node Deployment: 287990 (Finished)

PBS allocated job `287990.hunter-pbs01` at 2026-10-17 10:10:33 CEST, with
16 exclusive nodes and a deadline of 2026-10-18 10:10:33 CEST. Its bounded
deployment gate opened after all 44 Hunter tests and Bash syntax validation passed;
native worker launch was recorded at 10:15:23 CEST. All 64 worker log files exist.
The immutable source is `experiment192/source-sixteen-batch-20261017-v1`, archive
SHA256 `9609b38c1e0407ef2a29df65cbf8f696f4ac900ee3741f401bb315c773102600`.
Locally, 28 tests pass and the 16 native Bash cases are skipped.

This is a continuation of the six-node run, not a reset to parent step 328.
Arm A resumes step 530 (202 updates), and arm B resumes step 529 (201 updates).
FP32 model and Adam state, filtered sampler counters, best checkpoints, selection
windows, LR position and event history are retained. The original 1,000-update
budget is not restarted. Per-rank RNG is reseeded for the changed topology;
continuation is not bitwise identical. Both 32-worker backward/Adam hardware gates
and full recovery verification passed. Live history reached 214 updates in A and
213 in B, twelve fresh updates per arm. A job-bound inspection found four GPUs on
each of all 16 hosts, with 96-100% utilization in that snapshot. The exact deployed
source is backed up at GitHub commit `1f5f727e3df3ab772d01154e75d157e4984db0ea`.

The first new-job log backup is cloud-verified at
`5d31e936198ecf1c7a1893c486eb1c68967877e6`. During that run, the independent relay
uploaded every 300 seconds through the authenticated SSH tunnel. TensorBoard uses the same local
event mirror at http://127.0.0.1:16006. Keep the PC, VPN, tunnel and relay available.

## Previous Submission: 287977

The user requested release before resubmission, then changed the requested order
after deletion had been issued. PBS confirms the old six-node job finished with
exit status 137. Both arms had written final recovery markers at updates 202 and
201, and the final test was never opened. The old job's final logs were verified at
Hugging Face commit `86a93c132077a2094cc8c9e524414bf1d7156729`.

The following records describe the earlier six-node deployment and outage.

PBS accepted native batch job `287977.hunter-pbs01` and started it at
2026-10-17 09:25:09 CEST, with six exclusive nodes, 96 CPUs per node and a
24-hour deadline of 2026-10-18 09:25:09 CEST. The initial user-hold submission
was rejected by site policy and created no job; only the normal submission succeeded.
The source is `experiment192/source-six-batch-20261017-v1`, archive SHA256
`a1ce8b76594342a98ae9ea9917aac4b309f39acfe095cf03aeac9c37bb8931ba`.
All 31 tests passed on Hunter; 25 passed locally with six POSIX-only cases skipped.
GitHub verified this exact deployed bundle at
`d921f841dc43eefceff8b9bb40e505c3ccfac836`.

Native startup was recorded at 09:25:25 CEST. Subsequent inspection found 12
worker stderr files per arm and no repeated `pbs_attach` error. Hardware gates,
initial checkpoints, actual updates and GPU utilization were not yet verified.
The read-only GPU inspection did not run because its SSH shell lacked the
`HUNTER_WORKSPACE` export; this was a diagnostic-command error, not a trainer error.

Both authenticated Windows SSH connections then reset. No SSH processes or
18772/18773 forwarding listeners remained; port 22 on hunter-login03 was unreachable
from the current Wi-Fi route. No cached SSH agent or Windows VPN profile was
available. FortiClient was running, but connectivity had not been restored.
Do not infer the current PBS state from the cached running-state snapshot.
The native batch job is independent of the Windows SSH sessions.

The first new-job log backup is verified at Hugging Face commit
`3e427127f682fd6d1d437c35b0b76f0249aa2b1e`, snapshot 09:25:49 CEST.
The five-minute metrics relay remains configured to retry, but new cloud updates
require the authenticated tunnel. Local TensorBoard remains on port 16006 and
shows only its last synchronized data. The terminal allocation notification
returned to this conversation; evidence is in
`publication/notifications-287977/allocation.json`. The training-health notifier
is still armed, but also requires the forwarding endpoint.

## Automatic Launch

For the existing interactive job 287999, `control192.py --autolaunch` waits for
running state, actual start time and six allocated hosts. It validates exclusive
96-core resources and dispatches a single PBS-attached process on the master via
job-bound SSH. The master rechecks its hostname, PBS node file, immutable source
hashes and shared quota. Separate dispatch/start/run locks prevent duplicate launches.
The original `qsub -I` terminal must remain open; no replacement job is submitted.
The batch entry remains supported for future explicitly authorized batch requests.
All workers run natively through MPI; no nested per-worker attachment is used.
The actual PBS start determines the 24-hour deadline. Nonzero exits and missing
fresh updates are reported to the current-session notifier.

Training first runs its full-size backward/Adam gate on every worker, rolls that
update back, reloads the parent FP32/Adam state, and saves an initial recovery
checkpoint. The user authorized replacement allocations when necessary; retries
must first diagnose the error and verify the old job and its workers have stopped.

## Review Fixes and Storage Lifecycle

Both arms resume step 578 / new update 250 from job 287990, preserving full Adam,
sampler and selection state. The 32-to-12 transition reseeds rank RNG; same-layout
resume restores saved rank RNG. Neither is claimed bitwise-certified. History is
rolled back to the complete checkpoint and original events are purged above it.
Training-cutoff caching is tested draw-identical. Known-ID embedding rows are held
at resume values after Adam steps; the unknown row remains trainable. This prevents
unused-row drift but does not freeze the complete IDs-on reference score.

The first five-score selection window includes the baseline. Training draws are
not paired because weighted choice consumes RNG differently. Matched evaluation
uses 1,248 baskets per side, with 23 of 101 cells short of matched dates.
Compute-only, low-priority audits run alongside training; pruned historical ledgers
are reported unavailable. Final-test access requires scientific completion, never
only a deadline, quota or signal stop.

Startup requires six checkpoint payloads plus 8 GiB of headroom (about 103.6 GiB).
Runtime checks every ten updates reserve four payloads plus 8 GiB; each save checks
space for two concurrent payloads plus 8 GiB. Existing atomic-save checks remain.
Cloud recovery snapshots hardlink immutable checkpoint files; mutable logs are copied.
Log backups remain independent at 300 seconds. Private backups use job-scoped paths,
include complete stopped-run recoveries even after failure, and verify cloud hashes
and privacy before acknowledging cleanup on Hunter. Best/latest are always retained.
Superseded source checkpoints are eligible only after their own private receipt,
verified replacement receipt, finished allocation work and completed exposure audit.
The parent checkpoint, datasets, logs, code and metadata are never cleanup targets.
If backup fails, local recovery remains protected. This reduces quota-related risk;
it cannot guarantee against infrastructure failure or loss of the interactive shell.

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

The 16-node revision passes 28 local tests and all 44 tests on Hunter. Sixteen
native Bash tests exercise every arm/node-rank branch and post-module Socket transport.
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