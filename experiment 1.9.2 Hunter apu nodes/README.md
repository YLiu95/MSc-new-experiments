# Experiment 1.9.2

Two identity-hidden continuations from the verified Experiment 1.9 step-328 full
recovery. Job 287925.hunter-pbs01 requests 12 nodes for 24 hours. Each arm receives
six disjoint nodes and 24 GPU workers. Batch 1536 means 64 baskets per worker and
32 two-basket accumulation microsteps. Arm B alone changes market probabilities.

## Automatic Launch

The detached Hunter controller watches the existing user-owned PBS allocation.
It waits for running state plus valid start time and all 12 hosts, uses job-bound
SSH and pbs_attach, and never submits or deletes a job. Training first runs its
full-size backward/Adam gate on every worker, rolls that update back, reloads the
parent FP32/Adam state, and saves an initial recovery checkpoint.

Keep the original interactive qsub terminal alive. The controller cannot preserve
an interactive allocation whose original session is closed. Scheduler retries and
duplicate-start guards do not guarantee recovery from arbitrary runtime failures;
the monitor reports stalls for intervention rather than blindly duplicating workers.

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

The initial 15 training tests passed locally and on Hunter. The expanded suite
passed 18 tests on Hunter and 20 locally, including publisher isolation, cadence,
connection retry, stale-heartbeat detection and the post-module RCCL transport guard.
GPU hardware gates and actual first updates remain pending until PBS allocates nodes.

## Verified Setup

The active immutable training source is `experiment192/source-287925-transport`.
Watcher PID 979027 was verified alive on hunter-login03 with a fresh queued-state
heartbeat. Both arms still had zero updates at the recorded setup check.

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

`wake192.py watch` checks the existing private publication endpoint every 15 seconds.
Running state, a reported watcher failure, or a heartbeat older than three minutes
submits one new VS Code agent request through the supported `code chat --mode agent`
CLI. This is an actual prompt submission, not just a terminal completion notification.

The handoff contains the job identity, existing control-terminal ID and verification
requirements. It asks the agent to diagnose failures and continue until both hardware
gates and initial checkpoints are verified and both arms complete three real updates.
It prohibits duplicate training, allocation replacement, credential collection and
interruption of healthy workers, backups or the original allocation shell.

A persistent `dispatch.lock` prevents duplicate agent requests. The new agent must
write an acknowledgement matching the request UUID. State and evidence are stored in
`publication/wakeup/{watch-status.json,wake-status.json,agent-started.json}`.
An uncertain submission is reported rather than automatically duplicated.

A real harmless CLI probe successfully received acknowledgement for request
`c6cd13ca-8dc5-4870-ab3b-01d45fb4d4e3`; it only created its acknowledgement file and
did not access Hunter. Evidence is in `publication/wakeup-probe/`.

This starts a new agent request, not necessarily this same conversation. Keep this
workspace in VS Code, the Windows session awake, Copilot signed in, and the watcher,
VPN and SSH tunnel available. Existing tool approvals and authentication requirements
still apply; no approval or security setting is disabled. Wake-up does not guarantee
that every possible training failure can be repaired unattended.