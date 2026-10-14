# Experiment 1.7.1: Hunter MI300A

PBS/ROCm port of Experiment 1.7. Model: 10,193,843,857 parameters;
TP=4 within each node, DP across 1-4 nodes. Four GPU workers per node.
Global batch 320, microbatch 1, BF16 compute, FP32 weights/Adam state,
activation checkpointing, attention dropout 0.15 and identity dropout 0.10.
No padding, dropout relocation, compile, DeepSpeed or FP8 changes.

## Continuation Contract

The user approved a **warm start**, not exact continuation, from latest step 153
of `YL95/experiment-1.7-asc-gpu-nodes`, revision
`ecf683c5cfd3f49c9fbf2ce5e53f2f637dfce9b9`.
The source contains BF16 inference matrices and FP32 vectors only.
Optimizer, RNG and sampler are reset; samples from the earlier run may recur.
The learning-rate schedule continues at step 153. New scheduled-sample counts
start at zero. Subsequent same-layout Hunter recovery restores all local state.
TP8-to-TP4 conversion preserves packed Q/K/V and FiLM component order.

Data is `YL95/new_experiment_1-data` at
`bcbbefdbe2313673895eb1a0d354747a9f1624fa`.
Preparation retains the original imputation/label rules and chronological split.
Rebuilt vocabulary and primary validation manifest must match the parent digests.
Selection uses the frozen primary monitoring subset, not full validation/test.

## Hunter Setup

Set `HUNTER_WORKSPACE` to the allocated Lustre workspace. Stage source in
`$HUNTER_WORKSPACE/source`, dataset in `dataset/`, warm-start files in `warmstart/`.
Use the default documented PyTorch module and a workspace virtual environment:

```bash
module use /opt/hlrs/stack/ai-frameworks/modulefiles
module load pytorch/2.10.0_cray-python-3.11.7_rocm-6.4.1
python -m venv --system-site-packages "$HUNTER_WORKSPACE/venv"
source "$HUNTER_WORKSPACE/venv/bin/activate"
python -m pip install --no-index --find-links "$HUNTER_WORKSPACE/wheels" -r requirements.hunter.txt
source ./env.sh
python -m pytest tests -q
python -m ranking17.launch submit --mode preflight --nodes 2
```

The preflight allocation is two nodes for one hour, including offline panel
preparation if needed. The full model must pass backward, fused AdamW, save,
verified backup, bitwise restore-next-update and sampler-continuity checks.
Production reruns the full gate at its actual node count.

```bash
python -m ranking17.launch submit --mode production
```

Production selects four nodes if more than 20 APUs are free, otherwise two if
more than 10 are free. No request is submitted below that threshold. The
detached watchdog cancels queued requests after 600 seconds; no restart chain
is created. Deadlines use the running PBS allocation and are capped at
2026-10-15 08:00 CEST, with at least 20 minutes reserved for shutdown.

## Recovery and Publication

Compute jobs never call GitHub or Hugging Face. Recovery checkpoints have
digest-bound COMPLETE markers. Retention keeps latest recovery plus a verified
workspace backup, and best/latest allowlisted inference exports. Workspace
copies are not an independent failure domain. Quota is checked before saving.

The credential-local Windows publisher uploads verified inference exports,
aggregate reports and TensorBoard events to
`YL95/experiment-1.7.1-hunter-apu-nodes`. Raw panels, sampler ledgers, optimizer
state and tokens are excluded from public uploads. Hugging Face provides the
hosted Metrics/TensorBoard tab once event files are uploaded.
The publisher retries on a five-minute cycle; transfer duration is additional.
It needs Windows and the SSH tunnel to remain available. PBS training does not.

Source backups go to `YLiu95/MSc-new-experiments/experiment 1.7.1 Hunter apu nodes`.
Use `ranking17.publish source --root <local-report-dir> --credentials <dotenv>`
on Windows. The original experiment subtree is unchanged.

`ascgpu` and `ranker` remain vendored dependencies; their old launch/publication
entrypoints are not Hunter commands. Historical copied configs/reports and the
old requirements lock are not Hunter evidence and are excluded from publication.