# Experiment 1.9.2: Identity-Free Training With Holdouts

**Parent:** Experiment 1.9 step 328 full recovery checkpoint (private HF `YL95/experiment-1.9-hunter-recovery`,
revision `7a17855a414de2241469c1e3fd78f5bf351a69dc`, manifest
`133e851f4e4de23886b173f4c2874c8a9645e2531a185edde618256f607f2b01`).
**Basis:** [Experiment 1.9.1 diagnostic report](../experiment%201.9.1/DIAGNOSTIC_REPORT.md).

## 1. Why

In Experiment 1.9.1, training loss fell (0.686 → 0.642) while monitoring validation fell (0.075 → 0.059).
Diagnostics established:

| Finding | Evidence |
|---|---|
| The validation decline flows through stock-ID embeddings | IDs-hidden validation: 0.0818 (step 328) vs 0.0817 (step 5411) |
| IDs never helped on unseen years | Masking at step 328 cost only 0.0035 on validation (within ±0.004 noise) |
| IDs drive training-era memorisation | Training-era IDs-on score: 0.101 → 0.256 |
| The non-ID pathway also stopped generalising | IDs-hidden: training-era 0.079 → 0.099, validation flat |
| Small markets are over-exposed | Uniform market sampling: about 5,350 label reuses per NL stock-day vs 89 for the US |

Open question: is the remaining lack of validation progress memorisation through price histories, or
regime change after 2018? Existing checkpoints cannot answer it, so this run builds the test in.

## 2. Hypotheses

- **H1:** Training without stock IDs stops the validation decline.
- **H2:** Size-weighted market sampling improves validation further, mainly in small markets.
- **H3:** If validation stays flat without IDs, either (a) memorisation continues through price
  histories (detected by the stock holdout) or (b) the model has reached its ceiling on this data.

## 3. Design

Two arms run at the same time from step 328, each on 3 nodes (12 MI300As). They share the same base seed,
global batch, LR schedule, update budget and evaluation.

| Arm | Stock IDs | Market sampling |
|---|---|---|
| **A** | Always hidden (`identity_masked=True`) in training and evaluation | Unchanged (uniform over feasible markets) |
| **B** | Always hidden | Probability ∝ eligible training-period stock-days per market |

Reference curve: Experiment 1.9.1 (IDs on, constant LR 6e-5), same parent. Arm B differs from A only in
market-choice policy, but weighted choice consumes RNG differently, so dates and stocks are not exactly
paired between arms. Common fixed evaluation baskets, not training draws, support paired comparisons.

| Setting | Value |
|---|---|
| Global batch / microbatch / world per arm | 1536 / 2 / 12 (128 baskets, 64 microsteps per rank) |
| Update budget | 1,000 updates (1.536M samples) per arm, unless early-stopped |
| LR | Cosine from 6e-5 to 6e-6 over the budget; no warmup (Adam state preserved) |
| Optimizer / precision / dropout | Unchanged from 1.9.1 (AdamW wd 0.1, BF16 compute, FP32 master, dropout 0.15) |
| Sampler ledger | Restore parent 328 ledger (prevents exact-basket repeats) |
| Expected duration | Must be measured on three nodes per arm; the scheduler's 24-hour deadline is authoritative |

### Operational Amendments

Operational amendment, 2026-10-17: after six-node job `287977.hunter-pbs01`, the
user requested 16 nodes for 24 hours. Replacement job `287990.hunter-pbs01` uses
eight disjoint nodes and 32 GPU workers per arm with scheduler-derived deadlines.
Resume A from step 530 (202 updates) and B from step 529 (201 updates), retaining
FP32 model/Adam state, exact sampler progress, selection history and best checkpoints.
Global batch, LR schedule and the total 1,000-update budget remain unchanged.
Per-rank RNG streams are reseeded from the base seed, saved step and new world size;
different saved steps mean different arm streams. This is not bitwise continuation
of the former topology. Preserve old job artifacts; the final test remains sealed.
That run failed at update 300 because group quota could not hold the concurrent atomic saves.
The last complete recoveries for both arms are step 578 / 250 new updates / world 32.
Queued interactive job `287999.hunter-pbs01` requests six exclusive nodes with 96 CPU cores each,
for 24 hours. Resume both saved update-250 states on 12 workers per arm; reseed rank RNG for
32-to-12 migration, preserve Adam, sampler, best checkpoints and selection, and continue the
original 1,000-update budget. Old unsaved update-251-to-300 logs are not restored as progress.
Known stock-ID rows are restored to their resume values after every Adam step to prevent inherited
momentum or weight decay from moving these unused rows; the unknown-ID row remains trainable.
This operational amendment does not make the entire IDs-on reference score invariant.
Training-cutoff caching preserves the tested draw sequence; its throughput benefit is unmeasured.

Do not open new Copilot sessions or invoke Copilot CLI. Use terminal notifications
associated with the current conversation; idle-chat activation remains client-dependent.

## 4. Data Splits

| Split | Definition | Use |
|---|---|---|
| Train | Dates ≤ 2018-12-31 minus horizon, **excluding stock holdout** | Training only |
| Stock holdout | 10% of stocks per market, chosen by `sha256(market + ticker) < 0.1` | Never sampled in 1.9.2 |
| Validation | Existing 2019–2022 split and 90-session embargo; existing 1,616-basket monitoring subset | Model selection and early stopping |
| Final test | Existing `test` split: after 2022-12-31 plus 90-session embargo, to 2026-08-25 | **Sealed:** run once on finally selected checkpoints |

The parent saw all stocks, so the stock holdout is not clean in absolute terms. It is clean
**differentially**: included and held-out stocks share identical parent exposure, so any gap that opens
during 1.9.2 measures new memorisation.

Before use, confirm that no previous experiment evaluated on the `test` split.

## 5. Evaluation and Logging

| Cohort (8 primary tasks, 13 markets where feasible) | IDs | Frequency |
|---|---|---|
| Validation monitoring subset | Hidden (primary); on (reference) | Every 50 updates |
| Training-era 2015–2018, included stocks (1,248 baskets) | Hidden | Every 200 updates |
| Training-era 2015–2018, held-out stocks (1,248 baskets) | Hidden | Every 200 updates |
| Final test | Hidden | Once at end, for both arms' selected checkpoints and for step 328 |

Build training-era cohorts with the 1.9.1 diagnostic cohort builder. Use dates and tasks matched across
included and held-out stocks.

**Approved feasibility amendment (2026-10-16):** Keep the validation monitoring subset unchanged.
For included-vs-held-out training-era comparisons, use only jointly feasible market/task/date cells,
preserve the original basket sizes and 10% stock split, and report omitted cells and actual cohort
sizes. Never pad or duplicate stocks. The user confirmed that the final test split has never been
evaluated in earlier experiments; preserve that confirmation and keep access sealed until final
checkpoint selection is complete.
The frozen matched cohorts contain 1,248 baskets per side, not 1,616: 23 of 101 market/task cells
had insufficient matched dates. The preparation feasibility report records these omissions.

Per update, log what 1.9.1 logged (loss, gradient norm, score variance, LR, throughput) plus cumulative
label stock-day exposure per market.

Follow the experiment log-backup standard: public HF log repository, immediate upload and then every
300 s, no overlapping uploads, verified commits and a final flush. Optimizer, RNG, sampler, panels and
cohort definitions stay private.
After every stopped run, upload complete best/latest recovery payloads to the private recovery
repository and verify the pinned commit before cleanup. Retain the best 1B parent and current
best/latest per arm locally. Immutable snapshots use hardlinks, not full disk copies. Failed
uploads never authorize deletion; superseded runs require their own verified backup receipt.
Startup and runtime quota checks reserve space for both arms' concurrent atomic saves.

## 6. Stopping and Checkpoint Selection

- **Selection metric:** mean of the last 5 IDs-hidden validation scores (smoothed; SE about 0.002).
- **Initial window:** includes the pre-training baseline plus updates 50, 100, 150 and 200.
  Preserve this window and its saved history on resume; do not reset selection retrospectively.
- **Early stop:** no smoothed improvement > 0.002 for 6 consecutive validations (300 updates).
- **Retained checkpoints:** best smoothed and latest; keep the existing pre-expiry save, staged writes,
  SHA verification and rank-0 broadcast decisions.
- **Operational stops:** quota reserve, signal and deadline stops save recovery but do not freeze
  final selection or open the final test. Only budget completion or early stopping qualifies.

## 7. Decision Rules

Baseline: step 328 IDs-hidden validation **0.0818**. Its IDs-on 0.0853 is inflated by selection.

| Outcome | Interpretation | Next step |
|---|---|---|
| Smoothed validation ≥ 0.086 and stable, and final test ≥ step 328 on the same baskets | Changes work | Extend the winning arm to a longer, larger-budget run |
| Validation within ±0.004 of 0.082; training-era included-vs-held-out gap grows > 0.01 | Memorisation via price histories (H3a) | Reduce exposure: fewer passes, history augmentation, stronger regularisation |
| Validation within ±0.004 of 0.082; no gap growth | Ceiling on this data (H3b) | Stop scaling compute; change data, inputs or target |
| B beats A by > 0.004 smoothed, gains concentrated in small markets | H2 supported | Adopt size-weighted sampling |
| Validation declines | Unexpected | Inspect per market/task/year before any further run |

Report differences below 0.004 as inconclusive. Use paired time-block resampling for per-market claims.

## 8. Pre-Run Audits (CPU only, concurrent with training, do not block launch)

1. **Exposure audit:** count unique (stock, cutoff, horizon) labels and overlapping label-days used by
   Experiments 1.7, 1.7.1, 1.9 and 1.9.1 per market. Output: per-market reuse table.
2. **Universe audit:** survivorship (dataset is survivors only), point-in-time membership, delistings and
   label construction. Output: list of known biases affecting validation and test.
Run these at low priority on allocated compute CPUs, not login nodes. Historical checkpoint cleanup
removed the 1.7.1 and 1.9.1 ledgers; report these as unavailable, never reconstructed estimates.
Audit the retained 1.9 parent and both 1.9.2 update-250 filtered ledgers, excluding rejected parent
baskets. Completed processing does not imply complete historical coverage or external bias validation.

## 9. Implementation Tasks

Work in an isolated source copy (`experiment 1.9.2/`). Do not edit deployed 1.9.1 sources or the parent
checkpoint.

| # | Change | Location |
|---|---|---|
| 1 | `train192.py` from `train191.py`: `--arm {A,B}`, `WORLD=12`, budgeted cosine LR replacing `capped_learning_rate`, early stopping in the rank-0 decision path | `train191.py` |
| 2 | Pass `identity_masked=True` in the training forward (`update`) and in validation | `train191.py` `update`; `ranking17/train.py` `validate` |
| 3 | Sampler options: market weights (`generator.choice(possible, p=...)` restricted to feasible markets and renormalised) and held-out stock exclusion from `eligible` | `ranker/sampler.py` `Sampler._sample` |
| 4 | Deterministic stock-holdout list, hashed and recorded in the run config | New helper; config report |
| 5 | Cohort builders for training-era included/held-out and sealed test | Reuse 1.9.1 diagnostic code |
| 6 | Run two arms on one 6-node allocation (3 nodes each), separate artifact roots, locks and log-backup jobs | `run.sh`, launcher |

**Required tests before launch:**
- IDs are hidden in both training and evaluation (assert all `ticker_ids` map to the unknown ID inside
  the model).
- Market weights sum to 1, are renormalised over feasible markets and are deterministic per seed.
- Held-out stocks never appear in sampled baskets (property test over at least 10k draws).
- LR schedule endpoints and monotonic decay.
- Early-stop trigger on a synthetic score sequence.
- 1536-basket batch divides evenly across 12 workers.
- Parent manifest and payload verification.
- Full backward/Adam hardware gate on 12 GPUs per arm.
- Test-cohort access is blocked until the final evaluation flag is set.

## 10. Deliverables

- Source and tests in GitHub `YLiu95/MSc-new-experiments/experiment 1.9.2 Hunter apu nodes`.
- Public HF `YL95/experiment-1.9.2-hunter-apu-nodes`: logs, TensorBoard, inference exports, checkpoints.
- Results report: smoothed validation curves for both arms vs 1.9.1, training-era held-out gap,
  per-market effects, the final test table, the decision from Section 7, and audit outputs.
