# Experiment 1.9.2 Continuation: Finish Arm B and Run the Final Test

**Status (verified on Hunter 2026-10-17 16:58 CEST):** job `287999.hunter-pbs01` was an interactive
`qsub -I` allocation. PBS killed it at 14:45:27 CEST ("shepherd died from signal 9") after the client
session dropped.

| Arm | State | Checkpoints |
|---|---|---|
| A | Finished by early stop at update 650; selection frozen | Best step 928; latest step 978 |
| B | **Not finished:** last update 862 of 1,000; no `TRAINING_DONE.json`; selection not frozen | Best step 1078 (update 750, smoothed 0.0933); latest step 1178 (update 850). Checkpoint layout: 12 workers (3 nodes) |

The final test was **never opened**.

**Goal:** finish 1.9.2 under its own protocol, as a separate job from 1.9.3. Arm B completes its 1,000-update
budget, selection freezes, and the final test runs once. Experiment 1.9.3 starts only after these results
are reviewed.

## 1. Before Requesting Nodes (login node, no allocation)

Nothing below needs compute nodes. Finish all of it first, so nodes never wait for human work.

1. **Back up arm B.** Upload B's best (1078) and latest (1178) full recovery checkpoints and inference
   exports to the public HF recovery and model repositories. Verify size and hash against the cloud receipt.
2. **Record the test-split amendment** (before any test access; nobody has seen test results):

   | Split | Dates | Use |
   |---|---|---|
   | Test-report | After 2022-12-31 + 90-session embargo, to 2024-12-31 minus horizon | Opened by this continuation |
   | Final holdout | After 2024-12-31 + 90-session embargo, to 2026-08-25 | **Sealed** for the final model of the 1.9.x series |

3. **Code changes**, in an immutable source bundle with `READY.json` hashes:

   | # | Change | File |
   |---|---|---|
   | 1 | Single-arm B resume: same 12-worker layout, from step 1178. Restore FP32 model/Adam, sampler, selection state and per-rank RNG; budget unchanged | `train192.py` (resume path), `run.sh` |
   | 2 | Final selection reads arm A from the job-287999 root and arm B from the new root. Both require `TRAINING_DONE` with frozen selection | `evaluate192.py` |
   | 3 | Final test: test-report date filter; uses the allocation's world size (12), not `2 × WORLD`; scores IDs hidden (primary) and IDs on (secondary) | `final192.py` |
   | 4 | **Resumable final test:** fix the manifest at opening, save rows per checkpoint and ID mode, and finish only the missing items after an interruption. Never rebuild or reselect | `final192.py` |
   | 5 | Policy: `NODES=3`, `WORLD=12`, walltime `03:00:00` | `policy192.py`, `control192.py` |
   | 6 | Recovery relay uploads the latest verified checkpoint when PBS reports the job finished, even without `TRAINING_DONE` | `publish192.py` |

4. **Tests** (run on Windows and on Hunter before submission):
   - Final-holdout dates never appear in the test-report manifest.
   - The resumed B state matches step 1178: model/Adam hashes, sampler `issued`, selection window,
     `new_updates` = 850.
   - The budget stop at 1,000 writes `TRAINING_DONE` with `selection_frozen=true`.
   - The final test refuses to start until both arms are frozen, and an interrupted test resumes the
     missing rows only.
   - The worker count matches the 3-node allocation.
   - `bash -n run.sh`.

## 2. Node Request Strategy

**Request: 3 nodes, native batch job, 3-hour walltime.**

```bash
#PBS -N exp192_finish_B
#PBS -l select=3:node_type=mi300a:ncpus=96:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=03:00:00
#PBS -r n
```

Submit with `qsub run.sh` (no `-I`).

| Choice | Reason |
|---|---|
| **Batch, not `qsub -I`** | Job 287999 died because its interactive client session dropped. A batch job runs independently of the PC, VPN and SSH. |
| **3 nodes** | Matches B's saved 12-worker layout, so per-rank RNG state restores directly (no reseeding). Every node is busy for the whole job: B trains on all 12 GPUs, then the final test runs on the same 12. 6 nodes would save only about 15 minutes, add a non-bitwise reseed, and need more free nodes. |
| **3-hour walltime** | Estimated work is about 1–1.5 h: 150 updates at about 10 s each, plus 3 validations, plus 6 test evaluations. That leaves about 2× margin. Shorter jobs also fit scheduler gaps (backfill) more easily. The job ends and releases its nodes as soon as the script finishes. |
| **Submit only after all of Section 1 passes** | Nodes are never held while code is written or tested. |

The interactive 6-node job 288087 was cancelled before it started (17:04 CEST). No allocation is held now,
and none is needed until Section 1 is complete.

**Protection inside the job:**
- Checkpoint after every validation (every 50 updates), as now.
- PBS sends SIGTERM before the walltime kill; the trainer then saves and exits cleanly.
- The final test starts only if at least 1 hour remains. Otherwise the job records `EXPERIMENT_DONE`
  with `final_test_opened=false`, and a short follow-up job runs it.
- **If the job dies anyway,** resubmit the same script. Training resumes from the latest checkpoint; the
  final test resumes from saved rows.
- **Fallback, if B cannot be resumed twice:** freeze B at its verified best (step 1078, update 750) as a
  documented protocol deviation, then run the final test.

## 3. Job Sequence

| Step | GPUs | Estimate |
|---|---|---|
| 1. Verify source, prepared cohorts and the B checkpoint; 12-GPU hardware gate | 12 | About 5 min |
| 2. Train B, updates 851–1000 (validation at 900, 950, 1000; stock cohorts at 1000); budget stop freezes selection | 12 | About 30–35 min |
| 3. Final test on the test-report split, IDs hidden and IDs on: parent 328, A best 928, B selected | 12 | About 20–40 min |
| 4. Write `final-test/results.json`, `RESULTS.md`, `EXPERIMENT_DONE.json`; script exits and nodes are released | — | — |

Keep 5-minute public HF log backups running for the whole job. Afterwards, back up B's final checkpoints
and the final-test results.

## 4. Gate for Experiment 1.9.3

| Result (test-report, IDs hidden) | Decision |
|---|---|
| B selected ≥ parent 328, and B ≥ A | Proceed with 1.9.3 from B's selected checkpoint |
| B selected < parent 328 | Do not start 1.9.3. The validation gains reflect selection bias; revisit the design |
| B ≥ 328 but B < A | Size-weighted sampling is not confirmed; redesign 1.9.3's starting point |

Report differences below 0.004 as inconclusive, with paired 90-session time-block intervals. Report
IDs-on scores as a secondary inference option only.
