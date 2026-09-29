# Experiment 1.6: Task-Conditioned Financial Ranking

This private model card is the full technical documentation for Experiment 1.6.
The public Hugging Face root README intentionally contains only a link to this
file at a verified private GitHub commit. An unauthenticated reader may get 404.

## Status and intended use

As of 2026-09-29, the research pipeline and pinned validation suites exist,
with a corrected one-update GPU recovery smoke and an **invalidated** first
A/B/C screen, but no valid learning
screen result, held-out/task-transfer finding, test result, frozen
downstream probe, investment return, or profitability claim has been established.
Weights must not be presented as a calibrated expected return or probability.
Score ordering is a within-market preference for the specific $(L,H,K)$ and
comparison basket. Do not use this experimental system as financial advice.

The primary hypotheses are chronological within-market rank learning, the
value of explicit horizon/task conditioning on matched B/C streams, frozen
embedding transfer, and composition of seen L/H/K components in three excluded
task combinations. Conditioning may hurt some tasks; always report per-task
results and a size-matched fixed-task reference.

## Model, data and limitations

Inputs are 8-return nonoverlapping patches from 64 through 512 aligned market
sessions. A four-block width-256 temporal pre-LayerNorm Transformer pools a
history/identity/market embedding; an $(L,H,K)$ conditioner and two cross-ticker
blocks yield a contextual embedding and one unrestricted scalar score per
ticker. The base and contextual vectors are each 256-dimensional; ticker and
market identities, identity dropout, task embeddings and cross-ticker attention
receive end-to-end ranking gradients. The base embedding depends on history and
identity but not the requested horizon or comparison basket. Contextual
embeddings require the exact task and basket. The learned static lookup table
alone is not an encoder for new histories. A training-unseen identity maps to
the trained UNK row; use normal or identity-masked mode to inspect transfer.

Data uses private `YL95/new_experiment_1-data` at revision
`bcbbefdbe2313673895eb1a0d354747a9f1624fa`, `adj_close_clean`; its card
declares ODC-BY for its compilation and reports Yahoo Finance as its source.
It is **survivors only**: delisted firms' histories are largely absent, not
merely missing some future labels. The owner states they created the original
data and that the card's Yahoo attribution is inaccurate, and has authorized
the public A/B/C model backups. This discrepancy with the pinned card must be
resolved in the source documentation; neither account should be silently
treated as independently verified.
Retrospective adjustment and cleaning also preclude a point-in-time portfolio
claim. The source cache rejected 111,523 invalid price observations across
13 markets. Corrected 1.6 preparation excluded its 36,089 forward-filled,
flagged prices and both touching returns (61,505 finite returns removed).
Exchange calendars have not been independently certified. Training-period FP64
population standard deviation (ddof 0, no mean subtraction) is
`3.40288677815176` percentage points, fitted from 76,467,613 finite daily
returns dated through 2018-12-31. Inputs are divided by this value and clipped
to [-8,8]; labels are signed H-session sums accumulated in FP64 then cast to
FP32, with no clipping. There are 24,010 training-seen ticker identity rows and
one learned UNK row. Raw price panels, per-sample labels, basket identities,
eligibility lists and private predictions are not public artifacts.

Baskets are selected using historical validity only. A separate label mask
excludes missing future outcomes from loss/metrics without removing those
tickers from attention. This still leaves observed-outcome evaluation subject
to survivorship bias when delisting returns are missing. All tickers in a
basket share a market, cutoff, L, H, and K; signed returns rank winners above
losers, including negative returns. Each sample's observed, unequal pairs use
logistic loss at temperature 1 and are normalized by informative pairs before
global informative-sample normalization. A zero-informative global batch
advances sample attempts but not the optimizer schedule.

## Validation and selection

Train ends 2018-12-31; validation ends 2022-12-31; each split uses a 90-market-
session embargo and horizon-safe label endpoint. The frozen eight-task primary
validation manifest has 51,712 historical-eligibility baskets across 13 markets;
monitor has 1,616. The separate three-combination holdout has 19,456. The
registered selection criterion is eight-task-equal macro within-basket
Spearman, averaging baskets/date/market/task in that order, excluding constant
outcomes and reporting missing cells. The model checkpoint cannot be selected
by the monitoring subset, holdout, training loss, or test labels.

On the corrected primary suite, 3,344,940 label occurrences were observed
and 9,172 were unavailable; 123 baskets had degenerate Spearman and three
market/task cells were missing. Non-neural momentum macro Spearman was
+0.00105232, reversal -0.00105232, and constant scores 0. These are fixed-suite baseline
observations, **not** results for an untrained or trained neural ranker.
Pair accuracy counts tied predicted scores as half; top/bottom selections
report observed-outcome counts. No annualized Sharpe or portfolio claim is
supported without delisting treatment, transaction costs, and nonoverlap work.

## Recovery and reuse

The user authorized an eight-node, up-to-twelve-hour *screen request* with
A on two nodes (DP=16) and B/C on three nodes each (DP=24). The screen ends
after at most 2,000 updates per arm and does not continue into main training
without the user's separate review. These resource settings amend the initial
single-node proposal; they do not increase model capacity or change the
320-real-sample global batch.

Training schedules four distinct baskets per task visit, cycles feasible tasks
with seed 1337 and keeps exact run-wide stratum counters in a transactional
SQLite ledger. Reconnecting to an independently submitted Slurm job requires
its saved numeric ID; a restarted VS Code/Jupyter tunnel alone does not restore
the kernel. `best/weights.safetensors` contains BF16 matrix weights with FP32
normalization/bias and task/identity metadata; `latest/recovery.pt` contains
FP32 model and AdamW state and must be loaded only from a trusted source with
`torch.load(..., weights_only=True)`. `latest/sampler.sqlite`, rank RNG files,
training config, source digest and locked software versions are needed for
logical continuation on the same DP layout. Trained tensor and optimizer
states are not interchangeable with Experiment 1.5.

The owner requested three public repositories: A at
`YL95/experiment-1.6-asc-gpu-nodes-arm-a`, B at
`YL95/experiment-1.6-asc-gpu-nodes-arm-b`, and C at
`YL95/experiment-1.6-asc-gpu-nodes`. Each contains one arm's selected best,
distinct latest recovery state, training-seen vocabulary, aggregate reports
and TensorBoard events. If best and latest are the same checkpoint, public
`latest/` is omitted as requested; full private recovery remains in home
storage. A best at step 0 must be labeled an untrained selection, not a
learned improvement. Older blobs may remain in Git/LFS history. Public
weights, vocabulary and any distinct optimizer/sampler state are public
even though this explanatory card and raw corpus are private. No license
for this source/model is asserted by the backup operation.