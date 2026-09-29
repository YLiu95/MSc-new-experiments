# Experiment 1.6: Task-Conditioned Ranking on Athena

This is research software, not a trading strategy. The registered task ranks all
historically eligible tickers in a single aligned market basket for $L=64,72,\ldots,512$
returns, $H=1,\ldots,\min(90,L-1)$ sessions, and $K=3,\ldots,128$ tickers.
The historical decision record is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
The private [MODEL_CARD.md](MODEL_CARD.md) distinguishes measurements from proposed work.
The approved 8-node resource change is recorded in
[CAPACITY_AMENDMENT.md](CAPACITY_AMENDMENT.md).

## Private environment and prepared data

Use Python 3.11.5 in `$HOME/.venvs/experiment-1.6-gpu` and install
`requirements.txt` using the PyTorch CUDA 12.4 extra index. `requirements.lock.txt`
records the resolved installation. On this Jupyter account the home quota is 10 GiB,
so the home environment path is a symlink to an owner-only scratch-backed venv.
`env.sh` explicitly restores the cluster's Python/OpenSSL/SQLite runtime
libraries even in `--export=NONE` jobs while filtering conflicting
CUDA/NCCL/NVHPC paths. `scripts/runtime_preflight.sh` checks that bootstrap.
Do not source a dotenv file as shell code.

```bash
cd '/net/people/tutorial/tutorial042/projects/multi-step_forecast_MSc_project/new experiment 1/new_experiment_1.6_asc_gpu_nodes'
source ./env.sh
python -m pytest -q tests
python -m ranker.data --root "$ARTIFACT_ROOT" --source '/net/tscratch/people/tutorial042/experiments/experiment_1.5_asc_gpu_nodes/panel'
python -m ranker.evaluate register --root "$ARTIFACT_ROOT"
python -m ranker.evaluate baselines --root "$ARTIFACT_ROOT"
```

The source cache is accepted only when its dataset and revision exactly match
`YL95/new_experiment_1-data@bcbbefdbe2313673895eb1a0d354747a9f1624fa`.
Without that private cache, omit `--source` to download the pinned Parquet revision
using an authorized HF token loaded inside the download process. Raw returns,
eligibility indexes, evaluation basket manifests, and sampler ledgers stay under
`$ARTIFACT_ROOT`; never commit or upload those panels or per-sample labels.

## Registered screen and Slurm lifetime

The initial proposal was **at most 2,000 completed updates per arm and 48
GPU-hours aggregate**. The user's dated amendment authorizes an eight-node,
up-to-twelve-hour *request* (a 768 GPU-hour upper allocation ceiling) with A on
two nodes and matched B/C on three nodes each. This screen still stops at 2,000
updates per arm, releases all nodes after backup, and never automatically starts
main training. It uses a prefix of the full 30,000-update warmup/cosine schedule.
A is fixed-task; B/C share the same variable stream, with the conditioner
bypassed in B. Production keeps four historical-input-based samples per task
visit, 320 real attempts per effective global batch, eight replicas per node,
and four CPU data-loader workers per GPU; TP=1. B/C pad their last microstep
with 64 zero-weight execution samples that do not consume sample IDs.

Forecasts allocate **nothing**:

```bash
python -m ranker.launch forecast --hours 2
```

After reviewing the live forecasts and choosing `N` and the queue-wait cap,
submit explicitly, once:

```bash
python -m ranker.launch submit --nodes N --max-wait-seconds SECONDS --hours 2
```

For the approved amended screen, use `--nodes 8 --max-wait-seconds 60 --hours
12` only after its cross-node diagnostic and final dry run pass. The launcher
reduces walltime if 12 hours no longer fits before 08:00. A 12-hour request
is not a promise to hold nodes idle after the screen completes.

This submits with `sbatch` **without `--wait`**, records the numeric ID under
`$ARTIFACT_ROOT/control/job-ID.json` and `submissions.jsonl`, and uses Slurm's
`--deadline` so a job unable to complete after the chosen allocation-wait limit
does not start late. Recheck the reservation before every submission. The live
overnight reservation ends **2026-09-30 08:00 UTC+02:00**; the present JupyterHub
allocation ends **2026-09-29 23:31:41 UTC+02:00**. Losing the latter does not stop
an independent batch job. Job code calculates a conservative training stop from
the *earlier* of its actual EndTime and the reservation EndTime, verifies private
best/latest/log backups, and leaves at least 20 minutes before that earlier GPU
shutdown. Slurm also sends an early warning to the batch script.

On a **new** JupyterHub session, restart VS Code forwarding, source `env.sh`,
then inspect the retained job ID and files:

```bash
squeue --jobs ID -o '%i %j %T %S %e %R'
sacct --jobs ID --format=JobID,State,ExitCode,Start,End,Elapsed
tail -n 60 "$ARTIFACT_ROOT/control/slurm-ID.out"
ls "$ARTIFACT_ROOT"/{A,B,C}/checkpoints
```

Only completed checkpoints with verified `COMPLETE` markers can be resumed.
Reusing an arm's latest pointer via the node entry script preserves its optimizer,
RNG, sampler counters, and logical sample sequence. Lost work after the last
durable checkpoint can be recomputed; `restarts.jsonl` records that possibility.
The full primary validation suite, not the frequent monitor or held-out suite,
selects each arm's best weights.

## Artifact destinations

`python -m ranker.publish source --root "$ARTIFACT_ROOT"` pushes only code,
tests, scripts, environment lock, this guide, the private card, and both plan
formats to the **private** `YLiu95/MSc-new-experiments` repository under
`experiment_1.6_asc_gpu_nodes/`. It verifies branch, privacy, blob digests,
commit and remote tree without embedding secrets in Git URLs.

After the learning screen and a selected trained checkpoint exist, run
`python -m ranker.publish model --root "$ARTIFACT_ROOT" --arm C` (or the
explicitly selected arm) to publish **one** best inference export, **one** latest
full resumable state, the identity vocabulary, reconstruction metadata,
aggregate reports and that arm's TensorBoard events to
`YL95/experiment-1.6-asc-gpu-nodes`. The destination does not yet exist; the
publication command verifies account/dataset metadata and creates a public repo
only when complete trained artifacts and a verified private source commit exist.
Its root README is solely a link to the full private GitHub model card, so public
readers without GitHub access will not be able to read that card. A public full
optimizer/sampler checkpoint is trusted-source-only; do not unpickle untrusted
material. HF repository history may retain old blobs even when its head contains
only the selected pair.

For private TensorBoard, start `tensorboard --logdir "$ARTIFACT_ROOT/C/runs"
--host 127.0.0.1 --port 16007` on the current authorized host and forward the
port through VS Code. Do not publish a live tunnel or per-sample financial data.
The current Jupyter host serves named A/B/C and smoke runs on loopback port
**16007** (HTTP 200 verified); this service must be restarted after that
Jupyter allocation ends.

## Remaining research gates

The dated [WORK_PROCESS_REPORT.md](WORK_PROCESS_REPORT.md) records what has
actually been measured. Before a full 30,000-update comparison, complete the
GPU memory and throughput profile, reproduce a real-data resume and DDP check,
confirm the source calendar and redistribution rights, review the three learning
screens, and explicitly approve main-stage GPU time. Held-out combinations and
test data cannot choose the best checkpoint. An absent full comparison or frozen
linear probes must not be described as demonstrated forecasting skill.