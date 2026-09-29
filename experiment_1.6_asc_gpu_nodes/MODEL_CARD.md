# Experiment 1.6: Task-Conditioned Financial Ranking

This private model card is the full technical documentation for Experiment 1.6.
The public Hugging Face root README intentionally contains only a link to this
file at a verified private GitHub commit. An unauthenticated reader may get 404.

## Status and intended use

As of 2026-09-29, the research pipeline and pinned validation suites exist, but
no GPU training result, held-out/task-transfer finding, test result, frozen
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
declares ODC-BY. Retrospective adjustment, cleaning, unobserved delistings and
changes in ticker coverage limit any causal or portfolio interpretation. The
source cache rejected 111,523 invalid price observations across 13 markets;
new 1.6 preparation recorded no duplicate observations in that cache, but has
not independently certified each exchange calendar. Training-period FP64
population standard deviation (ddof 0, no mean subtraction) is
`3.4060786363283126` percentage points, fitted from 76,479,949 finite daily
returns dated through 2018-12-31. Inputs are divided by this value and clipped
to [-8,8]; labels are signed H-session sums accumulated in FP64 then cast to
FP32, with no clipping. There are 24,022 training-seen ticker identity rows and
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

On the primary suite, 3,352,558 label occurrences were observed and 1,554 were
unavailable; 99 baskets had degenerate Spearman and three market/task cells
were missing. Non-neural momentum macro Spearman was -0.00018994, reversal
+0.00018994, and constant scores 0. These are fixed-suite baseline
observations, **not** results for an untrained or trained neural ranker.
Pair accuracy counts tied predicted scores as half; top/bottom selections
report observed-outcome counts. No annualized Sharpe or portfolio claim is
supported without delisting treatment, transaction costs, and nonoverlap work.

## Recovery and reuse

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

The selected public repository is specified to hold one best and one latest
pair, the training-seen vocabulary, aggregate reports and approved TensorBoard
events. Older blobs may remain in Git/LFS history. Public weights, vocabulary
and optimizer/sampler state are public even though this explanatory card and
the training corpus are private. Dataset and derivative redistribution terms
must be confirmed before the first upload. No license for this source/model is
asserted without a rights review.