# Experiment 1.7: task-conditioned ranking

Research model with 10,193,843,857 parameters. It scores relative preferences within a same-market basket; scores are not calibrated expected returns. No portfolio, backtest, forecasting-improvement or test-generalization claim is made.

Architecture: temporal Transformer over eight-return patches, pooled base embedding, learned L/H/K FiLM conditioner, cross-ticker Transformer, shared scalar head. Width 5120; 20 temporal and 12 cross-ticker blocks; 80 attention heads. Checkpoints contain eight tensor-parallel shards. Instantiate `ranking17.model.RankingModel` with TP=8 and load the matching `weights-NN.safetensors` shard on each local rank. Shards must not be concatenated indiscriminately: attention QKV and FiLM output partitions have structured layouts.

Training uses pairwise logistic loss on masked forward log returns, with a chronological train/validation split and corrected imputed-price exclusion from Experiment 1.6. All task configurations are trainable. Best is selected using the frozen primary monitoring subset, not full validation. See per-run aggregate reports for measured progress; existence of weights is not evidence of outperformance.

Data lineage: YL95/new_experiment_1-data at revision bcbbefdbe2313673895eb1a0d354747a9f1624fa. The dataset declares ODC-BY and Yahoo provenance; the data owner has disputed that attribution. Data rights must be assessed independently before redistribution or commercial use. Raw data and per-sample observations are not included here. Dataset identity vocabulary remains in private prepared data; exact inference reconstruction requires that vocabulary and recorded normalization metadata.

Full resumable optimizer/RNG/sampler checkpoints remain in verified private storage. Public artifacts include selected best/latest inference weights and aggregate TensorBoard events only. Best-only publication means best and latest refer to the same update.

Source: https://github.com/YLiu95/MSc-new-experiments/tree/main/experiment_1.7_asc_gpu_nodes