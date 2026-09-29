# Experiment 1.6: Task-Conditioned Ranking and Reusable Financial Embeddings

Date: 2026-09-29. Status: research plan for discussion; no implementation, training, publication, or resource allocation is authorized by this document alone.

## 1. Scope and Decision Record

The primary research question is whether a shared, task-conditioned model learns useful within-market future-return rankings across different history lengths, horizons, and basket sizes. Secondary questions concern transfer between task configurations and the usefulness of the frozen encoder in other forecasting models. Larger model capacity is not the primary claim.

| Item | Status and decision |
| --- | --- |
| Current deliverables | Local Markdown and matching Word plan only. Implementation, training, and publication are future stages. |
| Input length | Interpret the user's correction as L = 8n market sessions, with integer n = 8 through 64 inclusive: L = 64, 72, ..., 512. This is 8 through 64 patches of 8 returns, not 8 through 64 returns. |
| Horizon | Integer H = 1 through 90 market trading sessions, strictly H < L. |
| Basket size | Integer K = 3 through 128, distinct tickers within one market and one aligned time window. |
| Objective | Whole-basket ranking, using pairwise logistic loss and primary Spearman rank correlation. User accepted the recommendation. |
| Embedding goal | Reusable inputs to other forecasting models; jointly learn from history, identity, and other legitimate available information. |
| Architecture below | Proposed engineering design, not a measured optimum or a claim of guaranteed transfer. |
| Sampling below | Recommendation in response to the user's question: scheduled task coverage, run-wide exact-sample nonrepetition, and fixed-budget reporting epochs. |
| Scientific comparisons | Smaller baselines, matched multitask comparisons, then conditional scaling. User accepted the staged recommendation. |
| GitHub destination | Intended private repository YLiu95/MSc-new-experiments, subdirectory experiment_1.6_asc_gpu_nodes/. Verify existence, privacy, and write access during execution. |
| Hugging Face destination | Public model repository YL95/experiment-1.6-asc-gpu-nodes. |
| Model card | Full card on private GitHub only; public Hugging Face README contains only a link to that private card. |
| Outstanding operating gate | Actual remote access, available GPU hours, storage policy, and execution approval remain unconfirmed. Proposed budgets below are ceilings, not reserved resources. |

The interpretation of the corrected input range is stated explicitly so it can be corrected before execution if necessary. Under this interpretation H = 90 is valid when L >= 96; at L = 64, H is limited to 63.

## 2. Evidence and What Experiment 1.5 Established

The supplied work-process report records a 7,444,254,721-parameter PyTorch model trained for 21 updates and 6,720 samples on 96 A100 GPUs. It did not complete an epoch. Its fixed-subset signed Huber loss was 5.023262 versus 5.004593 for a zero predictor. These are operational-pilot observations, not evidence of forecasting improvement or evidence that a billion-parameter model is needed.

The main job exited with failure after writing artifacts. A subsequent small distributed diagnostic exited cleanly, but a complete full-model resume-and-next-update equivalence test was not performed. Do not treat old checkpoints as directly compatible with the new objective, shapes, or sampler.

The supplied hardware inventory documents eight A100-SXM4-40GB GPUs per node, 40,960 MiB per GPU, NVLink connectivity within nodes, and Slurm-managed access. Inventory observations and scheduler predictions are historical, not current allocations. The report supersedes the older plan's JAX and TP=8/DP=1 defaults for describing what actually ran.

Exact Experiment 1.2 reference source was inspected at commit a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05. Its values are tabulated in Appendix A; this plan never delegates an experiment setting to an unspecified instruction to reuse 1.2.

## 3. Research Hypotheses and Interpretation

H1, primary: the conditioned multitask ranker improves chronological validation ranking over simple baselines and is competitive with a size-matched fixed-task ranker on that ranker's own task.

H2, conditioning: on an identical multitask sample stream, explicit task conditioning improves ranking over a model receiving variable-shaped histories but no explicit horizon signal. Shape alone identifies L and K but cannot identify H.

H3, representation: a frozen encoder improves downstream forecasts over a random frozen encoder and simple history features. Ranking performance alone does not establish reusable embedding quality.

H4, exploratory: a task combination omitted from training can be handled using components seen in other combinations. This is compositional generalization inside supported ranges, not extrapolation to unseen markets, horizons above 90, or histories above 512.

Different (L, H, K) settings are related conditional tasks under one ranking loss. Merely accepting different tensor shapes is insufficient evidence of useful multitask learning. Negative transfer is possible: shared parameters can help one horizon and harm another. Report task-wise results, not only a pooled number.

## 4. Task Space and Sample Definition

L belongs to {64 + 8j : j = 0, ..., 56}. H belongs to {1, ..., min(90, L - 1)}. K belongs to {3, ..., 128}.

There are 57 input lengths. The first four allow 63, 71, 79, and 87 horizons; the remaining 53 allow 90 each. Thus there are 63 + 71 + 79 + 87 + 53 x 90 = 5,070 length-horizon pairs and 5,070 x 126 = 638,820 task configurations before data eligibility and holdouts.

Reserve exactly these three task configurations for compositional evaluation: (128, 7, 16), (256, 21, 64), and (512, 90, 128). The main training task catalogue therefore has 638,817 entries before eligibility exclusions. Their individual L, H, and K values remain present elsewhere in training. Do not describe this as holding out entire horizons. The fixed-task reference (256, 7, 64) is not held out.

A sample is (dataset revision, market, cutoff t, L, H, sorted K ticker IDs). Its input contains returns at sessions t-L+1 through t, inclusive, for every selected ticker. Its labels cover t+1 through t+H. An input of L returns requires L+1 prices. No sample contains tickers with different cutoffs, histories from unrelated dates, or different horizons. A minibatch may contain multiple samples from different dates.

All tickers are candidates for ranking. Remove the old designated target and target-role marker. Ticker-order permutation changes storage order, not sample identity. Different H values are distinct supervised tasks even if their input arrays match; disclose that distinction in uniqueness counts.

## 5. Data, Labels, Missingness, and Leakage

### 5.1 Sources and normalization

Use dataset YL95/new_experiment_1-data at the recorded 1.5 revision bcbbefdbe2313673895eb1a0d354747a9f1624fa, conditional on that revision remaining accessible. A revision change requires a new data audit and run ID. Markets are AU, CA, CH, CN, DE, FR, GB, HK, IN, JP, KR, NL, and US. Use adj_close_clean, with its retrospective adjustment and cleaning limitations explicitly retained.

Daily returns are r(t) = 100 ln(P(t)/P(t-1)). The label is y(i,t,H) = 100 ln(P(i,t+H)/P(i,t)), equivalently the sum of H valid daily returns. Ordering log returns and simple returns gives the same ranking at a shared horizon. Never rank absolute returns: a large loss is not a winner.

Fit one population standard deviation on finite daily returns dated on or before 2018-12-31, across the 13 markets. Use FP64 statistics and ddof=0. Divide inputs by that standard deviation without subtracting a mean; clip normalized inputs to [-8, 8]. Preserve raw returns as FP32, normalized storage as FP16, and labels as FP32 computed with FP64 cumulative arithmetic before casting. Do not clip labels. Record the exact fitted scale after the audit; it is a data-derived measurement, not a configurable number that can be invented in this plan.

Reject duplicate market/ticker/session observations, nonpositive prices, and invalid calendar alignment with recorded reasons. Do not silently bridge missing market sessions or forward-fill prices to manufacture zero returns. Audit whether the market date union is a valid market-session calendar; do not assume missing sessions for the entire source are holidays.

### 5.2 Chronological boundaries

Retain train_end = 2018-12-31 and val_end = 2022-12-31. Let B_train and B_val be the last market-session indices on or before those dates. Set embargo_sessions = 90, replacing the old seven-session embargo.

| Split | Cutoff rule, in market-session indices |
| --- | --- |
| Train | t <= B_train - H |
| Validation | t >= B_train + 90 and t <= B_val - H |
| Test | t >= B_val + 90 and t + H is within the observed calendar |

All labels remain within their split's allowed endpoint. Input histories may extend into earlier periods because that information was known at the cutoff. The 90-session embargo is a conservative declared separation rule, not a claim that 512-session input overlap or market autocorrelation disappears. Purging label overlap and eliminating all statistical dependence are different goals.

Each task has its own history and endpoint eligibility. Do not reuse 1.2 or 1.5 anchor counts: changing L, H, and all-ticker supervision changes the indexed population. Record exact market/date/ticker/task counts and endpoint examples during preparation.

### 5.3 Future availability must not select the basket

Choose K distinct tickers using only complete historical input availability at t and legitimate as-of identity information. Do not require all K future labels to exist when selecting the basket. That requirement would favor survivors and silently discard delistings.

Keep input-valid tickers in cross-ticker attention even if a future label is unavailable. A separate loss mask excludes unavailable labels from comparisons; this mask must never be a model input. Require at least two observed, unequal labels for a ranking-loss contribution. An all-invalid or all-tied basket has zero ranking weight and an explicit counter; it is not silently replaced using future information.

Without verified delisting returns, masked-label evaluation is conditional on observed outcomes and remains potentially biased. Report missing-label rates by market, horizon, and ticker cohort. Do not assign zero or -100% to missing outcomes without an independently justified data source. Top/bottom diagnostics additionally report whether selected tickers have observable outcomes; never conceal unavailable outcomes by reselecting winners after seeing labels.

The public artifacts must contain no raw private panels, per-sample labels, eligibility lists, or private per-sample predictions. Aggregate metrics and approved model metadata can be published subject to dataset rights.

## 6. Ranking Objective and Metrics

The model outputs unrestricted real scores s(i), one per ticker. Scores are relative preferences, not calibrated expected returns or probabilities. Sort descending for inference. Do not train through a hard sorting operation.

For each unordered labeled pair i < j, let q(i,j) be +1 if y(i) > y(j), otherwise -1. Ignore pairs with exactly equal FP32 labels; tie tolerance is 0.0 percentage points. Pair loss is softplus(-q(i,j) x (s(i)-s(j)) / temperature), with temperature = 1.0.

Average pair losses within each sample over its informative pairs. Average sample losses over informative samples across the complete effective global batch. Do not average rank-local means equally when their valid counts differ. Use global sum/count normalization through gradient accumulation and data parallelism, with zero-weight execution padding. Abort an optimizer update with zero informative samples and log it without advancing the optimizer schedule.

K=3 supplies at most 3 pairs; K=128 supplies at most 8,128. Per-sample normalization prevents basket size from automatically multiplying loss weight. These pairs share tickers and are not independent observations. Log pair counts separately from unique samples.

| Metric | Exact convention |
| --- | --- |
| Primary | Spearman correlation of scores and realized returns within each evaluation basket, using average ranks for ties. |
| Degenerate correlations | Exclude baskets with constant realized returns and report their count; assign 0 to constant predictions when returns vary. Require at least 3 observable labels for primary Spearman. |
| Aggregation | Average baskets within market/date/task, then dates within market/task, then markets within task, then the eight registered primary tasks equally. Report missing cells; never silently change the selection denominator. |
| Pair accuracy | Informative observed-return pairs; score ties count as 0.5 correct. |
| Top/bottom diagnostic | Select q=max(1, floor(0.2K)) tickers at each end of scores; report mean realized log return and top-minus-bottom spread with observable-selection counts. |
| Portfolio claim | None. Overlapping-horizon returns and absent costs/delisting data do not constitute an investable backtest or annualized Sharpe ratio. |
| Baselines | Positive sum of the last L returns, its negative (reversal), and constant scores. Random ordering is a seeded diagnostic, not the only baseline. |

Use FP32 loss and metric reductions; accumulate reporting sums in FP64. Pairwise loss depends on score gaps, so do not interpret smaller loss as calibrated return accuracy. Check score variance, gradient norms, and tie rates for collapse.

## 7. Reusable Embedding Architecture

### 7.1 Recommended information flow

```text
Aligned return patches + recent-relative patch positions
             + ticker identity + market identity
                              |
                 Shared temporal Transformer
                              |
                  Mean pool + normalization
                              |
             Base embedding z(i): 256 dimensions
                              |
         Task conditioner (L,H,K) + cross-ticker Transformer
                              |
       Contextual embedding c(i;L,H,K,basket): 256 dimensions
                              |
                 Shared scalar ranking head
```

Both exported representations receive ranking gradients end to end. No stop-gradient separates the base encoder and ranker. Every informative pair trains the relevant score paths and shared layers; through attention it can also affect contextual inputs. No separate ticker classifier or auxiliary reconstruction loss is added merely to consume more signals.

The base embedding combines history and identity, but does not depend on requested H or the comparison basket. L affects it through actual history and positions. The contextual embedding also includes task and basket information. Export both, alongside the static ticker lookup table. The static table alone is not an embedding model for arbitrary new histories.

### 7.2 Exact initial model configuration

| Component | Value |
| --- | --- |
| d_model / exported dimensions | 256 / 256 |
| Attention heads / head dimension | 8 / 32 |
| Feed-forward width | 1,024 |
| Temporal blocks / cross-ticker blocks | 4 / 2 |
| Block style | Pre-LayerNorm Transformer, GELU feed-forward, linear biases enabled, LayerNorm epsilon 1e-5 |
| Attention | Bidirectional within observed history and within the current basket; no future tokens exist in inputs |
| Patch length / stride | 8 / 8; non-overlapping |
| Patch projection | Linear 8 -> 256 |
| Patch positions | 64 learned 256-dimensional positions; position 0 is the newest patch, consistently across L |
| Ticker identity | One 256-dimensional row per training-seen market/ticker identity plus one learned UNK row |
| Market identity | 13 learned 256-dimensional rows; no mixed-market baskets |
| Identity dropout | Replace a ticker ID with UNK with probability 0.10 per training sample/ticker, consistently across its patches |
| Standard dropout | 0.15 in attention probabilities and Transformer/feed-forward residual paths; disabled in evaluation |
| Pooling | Mean of valid temporal patch states, followed by LayerNorm; padding excluded |
| Ticker positions / role markers | Neither; ticker order must remain equivariant |
| Task embeddings | L: 57 x 16; H: 90 x 16; K: 126 x 16 |
| Task network | Concatenate 48 features -> Linear(48,128) -> GELU -> Linear(128,512) |
| Conditioning | Split task output into 256-dimensional gamma and beta; apply (1+gamma) x state + beta before the cross-ticker stack |
| Ranking head | LayerNorm(256) -> Linear(256,128) -> GELU -> Dropout(0.15) -> Linear(128,1) |
| Initialization | Linear weights Xavier uniform, biases zero; embedding/position rows normal std=0.02; normalization scale 1 and offset 0; final task-network layer zero-initialized |
| Activation checkpointing | Enabled for Transformer blocks |
| Computation / parameters | BF16 matrix computation, FP32 master parameters and optimizer moments |

The exact parameter count depends on the audited training vocabulary and must be generated from the instantiated model before execution. It is not the 296M reference model or the 7.44B pilot. A compact model is deliberately selected to test learning and representation quality with substantial data exposure.

Numeric task embeddings above support the registered finite settings and held-out combinations of their seen components. They do not promise smooth extrapolation to H=91 or an unseen patch length. A numeric-feature conditioner can be a later ablation, not an unrecorded change.

### 7.3 What 'no signal wasted' should mean

History, ticker identity, market identity, and the ranking objective can jointly shape an embedding. However, fitting all available detail can waste capacity or memorize ticker-specific training outcomes. Ranking also discards absolute magnitude information intentionally: getting +1% ahead of -1% and +20% ahead of -20% creates the same pair preference.

Consequently, ranking supervision alone may not make embeddings optimal for volatility, calibrated return magnitude, or entirely different assets. Start with one well-defined objective. Add an auxiliary return or volatility objective only if frozen-probe evidence establishes a deficit, with a separately registered weight and ablation. More losses are not automatically more information or better learning.

Ticker identity is useful but a memorization risk. Identity dropout trains the UNK path rather than leaving it random. Validation/test-only identities map to UNK; do not construct trainable rows from future labels. Export the same encoder in normal-identity mode and identity-masked mode; these are two modes of shared weights, not two independently trained models. Report both on unseen-identity cohorts.

Downstream consumers must know which embedding they receive. Base embeddings can be produced one ticker at a time; contextual embeddings require the exact history/task/basket contract. Never cache a contextual vector solely by ticker ID or reuse it across arbitrary baskets as if it were constant.

## 8. Sample Diversity: Recommendation and Teaching Notes

### 8.1 What expands and what does not

For a market/date/history length with N input-eligible tickers, a K-ticker basket has C(N,K) unordered possibilities. Horizons multiply labeled task views but can reuse identical input arrays. Nearby histories overlap, and every ticker/date/horizon outcome is reused in many baskets.

Example: changing one ticker in a 64-ticker basket makes a distinct input, but 63 histories and most pairwise labels remain the same. A one-session shift in a 512-session history shares 511 sessions. Exact uniqueness therefore measures data presentation diversity, not independent financial evidence.

The limiting resource may be independent dates and market regimes, not the number of possible tensors. Combinatorial augmentation can improve context robustness and reduce memorization of fixed baskets; it cannot manufacture new crises, regimes, or truly unseen outcomes. Larger sample counts must not narrow uncertainty estimates as if every basket were an independent observation.

### 8.2 Recommended nonrepetition contract

Prevent exact sample repeats throughout one logical run, including after resume. Allow the same raw history and label to appear in different tasks/baskets. Allow different experimental arms and seeds to reuse samples: matched examples are valuable for controlled comparisons.

The guarantee applies to the committed logical optimizer/sample sequence. A crash can require physically recomputing updates performed after the last durable checkpoint; those lost updates are not counted twice in the restored logical run. Preserve a separate restart/replay log instead of promising that no hardware operation ever repeats.

Define a reporting epoch as 3,000 completed optimizer updates x 320 sample attempts = 960,000 scheduled samples, excluding execution padding. Informative-loss samples are a separate count. Ten reporting epochs give 30,000 updates and 9,600,000 scheduled samples per main arm. An epoch is not a full pass over the astronomically large combination space.

Nonrepetition is an engineering constraint supporting diversity, not a scientific requirement for successful optimization. If limited strata exhaust or bookkeeping materially damages useful throughput, report that and revisit the policy rather than silently dropping coverage. Repetition can be beneficial and is not inherently leakage; overlap across chronological splits in future labels is the relevant leakage concern.

### 8.3 Reproducible sampler specification

Build the eligible task catalogue using historical-input availability, chronological endpoint rules, and N >= K; do not require observed future prices for every selected ticker. Publish aggregate feasibility counts privately; the mathematical task count is not a promise that all tasks exist in every market or date.

Shuffle the feasible training-task catalogue without replacement each coverage cycle using seed 1337. Each task visit schedules four distinct baskets. Within that task, select an eligible market uniformly, then an eligible cutoff uniformly within that market. This task-balanced policy prevents the US or long histories from winning solely by population size. It does not give each ticker or each L equal weight; report those induced weights. The longer L values have more valid horizons and thus more task entries.

At each stratum (market,t,L,H,K), keep a resumable counter c. Let C=C(N,K) for its sorted historical-eligible ticker list. Choose deterministic seeded integers b in [0,C-1] and a in [1,C-1] coprime to C, and map c to basket rank (a*c+b) mod C, then combinatorially unrank that rank. Handle C=1 separately. Stop using the stratum when c=C. This bijection guarantees no basket repeats within that stratum without storing every basket in RAM. The ordering is a structured seeded permutation, not a claim of a uniformly chosen permutation from all C! orderings; audit basket-overlap distributions.

Use arbitrary-precision integer arithmetic for binomial coefficients and rank operations. Persist touched-stratum counters, coefficients or reproducible derivation, task-cycle permutation/position, and assignment state atomically with the checkpoint. Keep counters in a transactional disk-backed index with bounded RAM caching. Benchmark unranking and indexing before adopting the sampler at scale. A failure to meet throughput or exact-restore tests is a gate, not permission to substitute ordinary random sampling while retaining a uniqueness claim.

Assign scheduled sample IDs centrally and distribute nonoverlapping blocks across data-parallel ranks. TP ranks, if later introduced, must share sample IDs. Worker count must not change the logical sample sequence. Physical padding, ticker permutation, dropout, and a different worker assignment do not create new sample IDs.

Each four-sample microbatch shares (L,H,K), avoiding padding across radically different tasks. Different microbatches in an accumulated global batch can use different tasks. Loss scaling is based on actual informative samples, not microbatch size alone. Do not choose only tiny K or short L because they are faster.

### 8.4 Coverage and dependency reporting

Log distinct supervised sample identities; distinct input identities excluding H; distinct market/cutoff pairs; unique ticker/date/horizon outcomes; task coverage-cycle progress; marginal L/H/K counts; per-market and per-ticker exposure; observed-label and informative-pair counts; and basket overlap diagnostics.

Report exact counters where the sampler establishes them. For very large distinct-outcome diagnostics, use explicitly labeled approximate cardinality estimates or offline exact audits; never label an estimate as a proven count. Chronological held-out performance and date-block uncertainty remain the evidence for generalization.

## 9. Comparisons, Training, and Compute Budget

### 9.1 Registered arms

| Arm | Training settings | Purpose |
| --- | --- | --- |
| B0 | No neural training; momentum, reversal, constant score | Establish a useful ranking reference |
| A | Fixed L=256, H=7, K=64; compact architecture above | Fixed-task learning reference |
| B | Same variable-task stream as C; task conditioner bypassed with gamma=beta=0 | Isolate missing explicit task information |
| C | Full variable-task training with explicit L/H/K conditioner | Main Experiment 1.6 candidate |

Instantiate the conditioner in all neural arms to keep parameter inventory aligned, but bypass its contribution in B. A uses its fixed task signal. Initialize shared weights identically per seed. B and C use the exact same scheduled samples and masks. A's different task exposure prevents a claim that A versus C isolates only conditioning; B versus C is the controlled conditioning comparison.

Compare A versus C on the (256,7,64) evaluation cell. B versus C use the eight-task primary suite. Report matched sample/update counts and actual GPU hours, because equal sample counts with different shapes do not imply equal compute or equal unique labels.

### 9.2 Optimizer and schedules

| Setting | Exact proposed value |
| --- | --- |
| Optimizer | AdamW; betas=(0.9,0.999), epsilon=1e-8 |
| Weight decay | 0.1 on linear weights and embedding/position tables; 0.0 on biases and LayerNorm scale/offset |
| Global scheduled batch | 320 real sample attempts; no execution padding counted |
| Microbatch per GPU | 4; reduce to 2 then 1 only after a recorded worst-case memory failure, preserving the global batch |
| One-node layout | TP=1, DP=8; gradient accumulation=10 at microbatch=4 |
| Maximum production updates | 30,000 per arm per seed |
| Reporting epochs | 10 x 3,000 completed updates |
| Warmup | 1,000 updates, linear from 0 to 2e-4 |
| Peak / final learning rate | 2e-4 / 4e-6 |
| Decay | Cosine from peak at update 1,000 to final at update 30,000; schedule never restarts at reporting-epoch boundaries |
| Gradient norm clipping | 1.0, global L2 before optimizer update |
| EMA | Disabled; best uses raw trained weights |
| Primary seed | 1337 |
| Confirmatory seeds | 1338 and 1339, only after the primary comparisons pass and budget is approved |
| Automatic early stopping | Disabled for matched-budget comparisons; best validation checkpoint still retained |
| Train scalar logging | Every 20 completed optimizer updates |
| Loss coefficients | Ranking=1.0; auxiliary regression=0.0; direction BCE=0.0; embedding-separation penalty=0.0 |

Use a real-sample count and an informative-loss count. If a scheduled global batch has no informative samples, log and consume those attempted sample IDs without an optimizer update; the 9.6M formula is nominal and the final report must state extra attempted samples from any skipped updates. Stop if 100 consecutive global batches have no informative labels; diagnose instead of training empty updates.

### 9.3 Staged operating ceilings

These are proposed planning ceilings requiring execution approval and scheduler availability. They are not claims of current access or enough time to finish all arms.

| Stage | Scope and ceiling |
| --- | --- |
| Correctness and profiling | Up to 8 GPU-hours aggregate; CPU preparation before GPU allocation |
| Learning screen | Up to 2,000 updates per A/B/C, seed 1337, and 48 GPU-hours aggregate, whichever limit is reached first |
| Main comparison | Up to 30,000 updates per A/B/C, seed 1337, and 192 GPU-hours aggregate, whichever limit is reached first |
| Additional seeds / capacity | Not included in the above authorization; proposed only after presenting learning, transfer, and timing evidence |

The screen is a prefix of the registered 30,000-update schedule, not a different short cosine schedule. Successful screened runs can resume without restarting or resampling. If GPU time prevents equal budgets, compare a common completed prefix and label unequal-exposure outcomes separately. Reserve 15% of each job's wall time for checkpointing, evaluation, and verified handoff, with a minimum 30-minute end-of-job reserve; require more when measured transfer time demands it.

Use one eight-GPU node per compact-model job initially. Extra nodes can host independent arms with isolated output paths. Do not automatically adopt TP=8 from the 7.44B pilot: communication cost is unlikely to be justified for this compact model. No larger architecture is a production candidate until a separate exact configuration, memory result, and budget amendment are recorded.

## 10. Validation, Selection, and Embedding Transfer

### 10.1 Fixed evaluation suite

Use these eight primary task cells: (64,1,3), (64,7,16), (128,21,32), (256,7,64), (256,63,64), (384,90,96), (512,7,128), (512,63,128). They are supported training configurations and span history, horizon, and basket size. They are a registered coverage suite, not a census of 638,820 tasks.

The three held-out configurations in Section 4 form a separate secondary transfer suite and never select the best checkpoint or trigger hyperparameter changes. Freeze them before training. Looking at them to redesign the model consumes their held-out status and requires a new held-out suite.

For each validation market/task cell, choose up to 128 cutoffs evenly spaced by ordered eligible cutoff index, determined without reading labels, and four deterministic baskets per cutoff, seed 7331. Maximum primary suite size is 13 x 8 x 128 x 4 = 53,248 baskets. The actual count and missing market/task cells must be audited and frozen before training. Require all eight tasks to have at least one evaluable market; otherwise amend the suite before training, never mid-selection.

For frequent monitoring, select up to 16 of those cutoffs and one of their four baskets per market/task: maximum 1,664 baskets. Evaluate monitoring every 500 updates; evaluate the complete fixed primary suite at step 0, every 3,000 updates, and at orderly final stop. Only the complete fixed primary suite selects the best checkpoint. This is full coverage of the declared benchmark, not all validation anchors or all possible baskets.

Select the highest primary macro Spearman; an exact tie retains the earlier checkpoint. Do not select on pairwise training loss, the monitoring subset, test results, or the held-out combination suite. Each experimental arm keeps its own local best for comparisons; public export ultimately contains only the selected final model's best/latest pair.

After freezing architecture, settings, and selection, evaluate the selected best models on a test suite constructed by the same rule with seed 9331. No test-driven retraining. Retain full chronological date/ticker exclusion reports privately.

Report paired model differences with a moving-block bootstrap: 128 consecutive dates in the union of all 13 market-session calendars, 2,000 resamples, seed 2337, 95% percentile interval. These are actual calendar-session positions, not 128 sparsely sampled evaluation cutoffs. Use the same sampled date blocks for all markets, tasks, and compared models; include every recorded evaluation observation whose date falls inside each block, and retain observation multiplicity when a block is drawn more than once. Preserve the registered macro aggregation. This is an approximate dependence-aware diagnostic, not proof of independent markets or guaranteed interval calibration. Verify that each cell spans at least two disjoint blocks with observed evaluations; otherwise report insufficient temporal coverage. If resampling produces an empty required task cell, mark that replicate invalid and report its count rather than silently changing task weights. Three training seeds, when available, address a different source of variation.

### 10.2 Frozen downstream forecasting probes

Evaluate representation reuse separately from the ranking head. Freeze encoder parameters; fit downstream ridge regressors for signed H-session return at H in {7,21,63}, using L=256 and individual-ticker base embeddings. No comparison basket is needed. Also evaluate contextual embeddings using fixed K=64 baskets and the matching H, clearly labeling their additional context requirement.

Compare trained normal-identity embeddings, trained identity-masked embeddings, a random frozen encoder with seed 1337, and simple features: means and population standard deviations of the last 8, 32, 64, 128, and 256 returns (10 features). Fit feature standardization on probe-training data only. Ridge fits an intercept; choose alpha from {0.1,1,10,100} on validation MAE and then freeze it for test. Use the same examples for competing feature sets and report chronological MAE, RMSE, and cross-sectional Spearman; never compare on different missing-label populations.

Cap each probe split at 100,000 train, 20,000 validation, and 20,000 test anchors, sampled reproducibly with seed 4337, equal-market quotas redistributed only when a market lacks sufficient anchors. Select using historical eligibility and split endpoints; disclose missing-label exclusions. These probes reuse the selected ranking checkpoint without encoder fine-tuning and do not select or retrain that checkpoint. Held-out task-combination results are not inputs to probe fitting.

A successful frozen linear probe supports accessible predictive information, not proof that every downstream model benefits. Future work may test a downstream nonlinear model or fine-tuning, but those are separate experiments with their own capacity and leakage controls.

## 11. Athena Execution and Software Contract

No commands in this plan have launched training. At execution, recheck Slurm reservations, job limits, node health, CPU/RAM entitlement, quota, and Jupyter lifetime. The recorded overnight training reservation ends 2026-09-30 08:00 UTC+02:00; training-multigpu-2 was scheduled 09:00-16:00 that day. Neither window guarantees personal GPUs or access after its end. The one-hour gap is not assumed usable.

Initial job request: 1 node, 8 GPUs, 32 allocated CPUs, 256 GiB host RAM, allocation-wait limit 60 seconds, and a wall-time selected within the currently verified authorization. Start one torchrun agent on that node with 8 workers. Each GPU hosts one compact-model replica. Use independent sbatch jobs and persistent rank logs; do not depend on the local Windows terminal, notebook kernel, tmux, or the Jupyter allocation surviving.

Proposed pinned direct runtime dependencies: Python 3.11.5, torch 2.6.0+cu124, numpy 2.2.6, pandas 2.2.3, pyarrow 19.0.1, huggingface-hub 0.30.2, tensorboard 2.19.0, safetensors 0.5.3, python-dotenv 1.1.0, scipy 1.15.2, scikit-learn 1.6.1, and pytest 8.3.5. These are explicit engineering candidates, not a verified installation or a claim they equal the historical 1.5 lock. Resolve and security-check the full transitive lock before execution; any required version change is documented in an amended resolved config, never described as already validated here.

Use a private environment under `$HOME/.venvs/experiment-1.6-gpu` and scratch root `/net/tscratch/people/<actual-user>/experiments/experiment_1.6`. Discover the actual username; do not assume tutorial042 remains valid. Keep raw data, panels, sampler indexes, runs, and recovery artifacts off Git. Preserve Python/OpenSSL runtime paths when filtering competing inherited CUDA library paths. Clear conflicting inherited Slurm memory variables only for each submission.

Use four data-loader workers per GPU, bounded prefetch of two microbatches per worker, pinned host-memory transfers, and memory-mapped panels. Benchmark these CPU/I/O settings before production; do not preload all combinations. No compiler or CUDA graph requirement is imposed initially. Use PyTorch scaled-dot-product attention where supported, validating masks and dropout behavior.

Profile actual inputs at L=512, H=90, K=128 as well as small cases. Warm up 10 updates and measure 50 updates, including optimizer state allocation and checkpoint staging. Collect allocated and reserved peaks on every GPU. Target reserved memory <= 34 GiB per 40-GiB GPU; reduce microbatch according to Section 9 if exceeded. Check finite gradients and meaningful score variation, not only device visibility.

## 12. Correctness and Research Gates

Before full training, implement and pass the following focused tests:

1. Enumerate 57 L values, 5,070 valid (L,H) pairs, and 638,820 tasks; reject H>=L, H>90, L outside the corrected range, and K outside 3..128.
2. Verify return labels and ordering on known prices, including negative returns and exact ties; labels use no absolute-value transformation.
3. Verify every split boundary and H=90 endpoint; no label crosses a split boundary, and L=512 requires 513 prices.
4. Verify basket membership is unchanged when future availability is altered; unavailable future labels affect loss masks, never inputs.
5. Exhaustively test small combinatorial domains for nonrepetition, bijection, exhaustion, and deterministic resume; test arbitrary-precision domains without integer overflow.
6. Verify ticker permutation equivariance, temporal padding exclusion, per-sample loss normalization, prediction-tie metrics, and zero-informative-batch behavior.
7. Verify nonzero gradients into temporal, identity, market, cross-ticker, and task pathways after the zero-initialized conditioner begins learning; check the UNK training path.
8. Verify base embeddings are unchanged by H or basket when history/identity are fixed, and contextual embeddings respond to the requested task when learned.
9. Compare a one-GPU reference and DDP outputs/gradients with uneven valid-label counts and execution padding. Use FP32 relative tolerance 1e-5 and absolute tolerance 1e-6; BF16 relative/absolute tolerances 0.02/0.02 for the registered small diagnostic, reporting maximum errors.
10. Stop and restore a short real-data run; match next sample IDs, labels, counters, optimizer schedule, and next update against uninterrupted training. Bitwise identity is not promised across nondeterministic GPU kernels; record tested tolerances and layout.
11. Validate best export round-trip, full latest-state recovery, complete-manifest detection, interrupted-upload behavior, and public-upload allowlisting.
12. Confirm model selection uses the complete registered primary validation suite only. No test labels or held-out combination performance may choose checkpoints.

A screen must show declining train loss on a small overfit fixture, noncollapsed scores on real data, complete monitoring output, and valid checkpoint recovery before main-run approval. Baseline superiority on a small screen is not required to prove final skill, but persistent failure or data defects must be investigated before increasing capacity.

## 13. Checkpoints, TensorBoard, and Later Publication

### 13.1 Local recovery

Write rolling recovery checkpoints every 30 minutes and at orderly stop, keeping the latest two complete local recovery versions until their replacements are verified. Save best weights whenever the complete registered primary validation score improves. Save each experimental arm in a separate directory.

Latest includes FP32 model parameters, AdamW state, learning-rate position, RNG states, sampler task-cycle and stratum state, issued/consumed sample state, sample/update counters, next evaluation and checkpoint thresholds, best metric, dataset revision, scaler, training identity vocabulary, exact model/task config, software lock, and parallel layout. Training must quiesce prefetched/in-flight sample assignment at checkpoints or save it explicitly; saving only the optimizer is insufficient for nonrepetition on resume.

Best is an inference export including base encoder, task conditioner, cross-ticker encoder, ranking head, trained identity tables, and input/output reconstruction metadata. Use safetensors with BF16 matrix weights and FP32 normalization parameters; run exported-output tolerance checks. Preserve latest in full resumable precision. Training optimizer pickle-style payloads are not needed for inference and must be documented as trusted-source-only artifacts in the private card.

Write into staging directories, finish and hash all files, test readability/restore, then atomically publish a completion marker and move best/latest pointers. Never remove the last verified local backup because an upload command merely returned success. Checkpoint export/restoration must not gather unnecessary full tensors onto GPU 0.

### 13.2 TensorBoard

Serve TensorBoard privately on the current authorized host, bind 127.0.0.1, proposed port 16007; if occupied, use the next available port and record it. Use authenticated forwarding, not an unapproved public tunnel. The public event files are an explicit publication decision separate from private live service access.

Log train loss/LR/gradient norm, raw score variance, validation task and market metrics, baseline differences, embedding norms, missing-label rates, identity exposure, sample/input/outcome coverage, throughput, GPU/host memory, checkpoint duration, and dependency versions. Do not dump the environment or sample-level financial records. Flush and upload only approved aggregate events. File presence alone does not establish a working hosted TensorBoard viewer.

### 13.3 Destination contract

Private GitHub destination: https://github.com/YLiu95/MSc-new-experiments, folder experiment_1.6_asc_gpu_nodes/. Include source, tests, exact configs, environment lock, launch/resume documentation, both plan formats, research diary, aggregate reports, and MODEL_CARD.md. The unauthenticated URL returned 404 during planning; privacy, existence, default branch, and write scope must be verified later.

Public Hugging Face destination: https://huggingface.co/YL95/experiment-1.6-asc-gpu-nodes. Keep one selected model's best inference directory, one latest resumable directory, necessary reconstruction metadata, aggregate reports, and approved TensorBoard events at the repository head. Intermediate arm checkpoints stay on approved private storage. Best and latest may reference the same update but serve different purposes. Repository/LFS history may retain older blobs; this policy does not claim historical erasure.

Do not publish the full model card to Hugging Face. Its root README.md, which Hugging Face treats as the model card, contains only a Markdown link to the private GitHub MODEL_CARD.md at a verified commit on the actual default branch. Do not add copied card text or descriptive YAML frontmatter. This minimal link file is the explicit exception requested by the user, not an additional public full card. Public users without GitHub access will not be able to read the linked documentation; they may see 404. A private link does not make the public weights, vocabulary, recovery state, or logs private, nor does it prevent memorization-related privacy risks.

Verify repository visibility and dataset/model redistribution rights before first upload. Check all remote file sizes and digests against the local manifest and verify the GitHub commit. Do not publish a repository license that has not been chosen and checked against data and dependency obligations.

### 13.4 Credentials

The user identified local credentials at C:\Users\YL\dev\.env under GITHUB_TOKEN and HF_TOKEN. They are not needed to write this local plan, and were not read for this task. During future authorized execution, load only required variables inside the process making authenticated requests. Never print, copy into the plan, stage, or serialize the values. Do not execute dotenv files as shell code.

The local Windows file does not automatically exist on Athena. Arrange an authorized remote secret mechanism or a private remote dotenv file outside the checkout with directory mode 700 and file mode 600. Do not transmit secrets through chat, Git URLs, command-line flags, logs, checkpoints, or TensorBoard. Separate data-download and publication permissions where practical.

## 14. Decision Gates Before Execution

1. Confirm the explicit interpretation L=64..512 in steps of 8, and accept or amend the proposed sampler and embedding architecture after discussion.
2. Confirm an actual GPU-time ceiling and remote access. Approve a stage before launching it; the plan itself makes no allocation.
3. Audit the pinned data, calendars, scaler, feasibility, missing labels, and training-only identity vocabulary; freeze exact measured values and evaluation manifests.
4. Validate the complete software lock, tests, worst-case throughput/memory, sampler overhead, checkpoint round-trip, and shutdown behavior.
5. Approve the main comparison after reviewing the learning screen. Additional seeds, auxiliary objectives, or larger models require an explicit documented amendment.
6. Freeze selection before test evaluation. Verify private/public destination policies and rights, publish the requested best/latest pair and logs only during the later authorized execution stage.

The remaining uncertainty is empirical, not permission to leave numerical settings unspecified. Values such as exact audited sample counts, fitted scale, vocabulary size, instantiated parameter count, measured throughput, job wall time, and remote default branch must be recorded once measured. They cannot honestly be supplied as known figures today.

## Appendix A. Exact Experiment 1.2 Reference Values

These are source configuration defaults at the pinned commit, not proof that every historical invocation used unmodified defaults and not automatic 1.6 settings.

| Reference setting | Exact 1.2 source default |
| --- | --- |
| run_name / seed | patchtst_global_tpu_v1_2 / 1337 |
| dataset_repo / price_column | YL95/new_experiment_1-data / adj_close_clean |
| Markets | AU, CA, CH, CN, DE, FR, GB, HK, IN, JP, KR, NL, US |
| n_steps_in / patch_len / patch_stride | 256 / 8 / 8 |
| horizon / n_tickers_per_sample | 7 / 64 |
| return_clip | 8.0 |
| train_end / val_end / embargo_sessions | 2018-12-31 / 2022-12-31 / 7 |
| d_model / n_heads / d_ff | 1024 / 16 / 4096 |
| temporal_depth / cross_ticker_depth | 12 / 8 |
| dropout / remat | 0.15 / True |
| magnitude_loss_weight / direction_loss_weight | 0.7 / 0.3 |
| magnitude_huber_delta_pct | 1.0 |
| epochs / steps_per_epoch / val_batches | 60 / 500 / 80 |
| batch_size | 320 |
| learning_rate / min_lr_fraction | 2e-4 / 0.02; implied floor 4e-6 |
| weight_decay / warmup_epochs | 0.1 / 2; implied warmup 1,000 updates |
| gradient_clip / ema_decay | 1.0 / 0.999 |
| early_stop_patience / log_every_steps | 12 epochs / 20 steps |
| backup_every_epochs / keep_last_n_checkpoints | 5 / 2 |
| hf_repo_id | YL95/new_experiment_1.2_tpu |
| github_repo / github_subdir | YLiu95/multi-step_forecast_MSc_project / new experiment 1/new_experiment_1.2_tpu |
| Default artifact_root | /root/artifacts/new_experiment_1.2_tpu, overridable by ARTIFACT_ROOT |
| loss_weights_path | Experiment-root loss_weights.json |
| Mag 7 list | AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA |

The 1.2 nominal epoch was 500 x 320 = 160,000 samples; its 60-epoch maximum implied 30,000 optimizer updates and 9,600,000 sampled examples, not exhaustive anchor coverage. The 1.6 production update ceiling matches that numerical reference but changes task distribution, objective, epoch reporting, architecture, EMA, and validation. It is not a matched reproduction.

The inspected 1.2 preparation source fits global training population variance as sum_squares/count - (sum/count)^2 and divides by its square root without centering inputs. It writes raw returns to FP16; 1.6 explicitly retains FP32 raw returns to reduce label quantization and tie artifacts. Source split masks are train t<=B_train-7, validation B_train+7<=t<=B_val-7, and test t>=B_val+7. Section 5 replaces those horizon and embargo constants explicitly.

The original model adds ticker identity and target-role embeddings before temporal attention, pools time, applies cross-ticker attention, and conditions two heads on the selected target state and identity. Section 7 retains identity-aware encoding while removing target roles and replacing two target-only outputs with shared per-ticker scores.

Exact original requirements file: jax==0.10.2, flax==0.12.7, optax==0.2.8, numpy==2.5.0, pandas==3.0.4, pyarrow==24.0.0, huggingface-hub==1.21.0, tensorboard==2.20.0, tensorboardX==2.6.5, setuptools<81, pytest==9.0.2. This is a historical source declaration; it is not a GPU compatibility prescription or a fully pinned transitive lock. The explicit 1.6 runtime proposal is in Section 11.

## Appendix B. Sources and Verification Scope

Supplied local source documents, read in full for this planning discussion:

- experiment 1.5 WORK_PROCESS_REPORT.md, under the supplied deep learning experiments directory.
- EXPERIMENT 1.5_PLAN.md, same directory; historical defaults distinguished from its execution update.
- asc gpu_environment.md, same directory; includes the later reservation and Jupyter-lifetime observations.

Pinned public Experiment 1.2 sources inspected:

- https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05/new%20experiment%201/new_experiment_1.2_tpu/src/config.py
- https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05/new%20experiment%201/new_experiment_1.2_tpu/src/model.py
- https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05/new%20experiment%201/new_experiment_1.2_tpu/src/prepare_data.py
- https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05/new%20experiment%201/new_experiment_1.2_tpu/src/train.py
- https://github.com/YLiu95/multi-step_forecast_MSc_project/blob/a8e7a7dd55424b6c4d8de0553f9a1f43539a4b05/new%20experiment%201/new_experiment_1.2_tpu/requirements.txt

This planning task did not download the private dataset, inspect secret values, test the proposed GPU stack, run an experiment, authenticate to either destination, or publish anything. Document checks validate the plan's internal arithmetic and Word packaging, not the scientific hypotheses or executable training implementation.