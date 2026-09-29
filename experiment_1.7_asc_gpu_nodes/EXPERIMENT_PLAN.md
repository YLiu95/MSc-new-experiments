# Experiment 1.7: >10B Task-Conditioned Ranking (built from Experiments 1.6 and 1.5)

## Purpose

  

Modify the code of Experiment 1.6 and Experiment 1.5 to create and run Experiment 1.7: train a ranking model, multi-task in the "many conditional instantiations of one loss" sense as in Experiment 1.6, at a size above 10B parameters, at the execution scale of Experiment 1.5.

  

## Objective and task definition (from Experiment 1.6)

  

- Objective: whole-basket ranking with a pairwise logistic loss on score gaps; primary metric is within-basket Spearman rank correlation. Scores are relative preferences, not calibrated expected returns, and no portfolio or backtest claim is made.

- Multi-task means many conditional instantiations of one ranking loss. A training task is the triple (L, H, K): history length, forward horizon, basket size. Tasks are given to the model through explicit learned L/H/K conditioners (FiLM gamma/beta), because tensor shape identifies L and K but cannot identify H.

- 1.6 task space: L = 64, 72, ..., 512 (57 values); H = 1 ... min(90, L-1); K = 3 ... 128, giving 638,820 configurations; do not reserve (128,7,16), (256,21,64), and (512,90,128) evaluation. 

- A sample is (market, cutoff t, L, H, K, sorted K tickers). Inputs are the L returns ending at t; labels are the forward H-session log returns. Unavailable future labels are excluded by a loss mask and never used as model inputs or to select the basket.

- Reuse 1.6's data pipeline, chronological splits, and sampling contract unchanged unless an explicitly recorded amendment says otherwise.

  

## Model and scale (1.5 execution, above 10B)

  

- 1.5 reference: PyTorch, a measured 7,444,254,721-parameter model trained on 12 nodes with TP=8 per node and DP across nodes for 21 updates; it completed no epoch and did not beat its zero-return baseline. 1.7 widens the same code lineage past 10B and must record the instantiated parameter count before launch.

- Keep 1.6's architecture family: temporal Transformer, pooled base embedding, task conditioner plus cross-ticker Transformer, contextual embedding, shared scalar ranking head.

- BF16 matrix computation with FP32 master weights and activation checkpointing; measure worst-case memory and throughput before full training. TP=8 within node plus DP across nodes is required at this size.

- Do not warm-start from 1.5 checkpoints: they are not compatible with the 1.6 objective, shapes, or sampler.

  

## Execution, shutdown, and backup

  

- Request 8 nodes, or as many as are available, for 10 hours, or as many hours as possible, via sbatch with one torchrun agent per node: 8 workers, one model replica per GPU, persistent per-rank logs. Recheck the reservation before every submission and cap walltime so the job fits before its end rather than starting late.

- Run training until 20 minutes before GPU shutdown, then back up. The established rule computes the training stop from the earlier of the job's actual EndTime and the reservation EndTime, lets Slurm warn the batch script, and leaves at least those 20 minutes for checkpointing and verified backup.

- Write rolling recovery checkpoints during the run and a best checkpoint whenever validation improves. Resume only from checkpoints carrying a verified COMPLETE marker, preserving optimizer state, learning-rate position, RNG state, sampler counters, and the logical sample sequence; record every restart.

- Verify checkpoint, backup, and restore before trusting a long run: 1.5's main job exited with failure after writing artifacts, and a full resume-and-next-update equivalence test was never completed.

  

## TensorBoard

  

- Serve on port 16006, the Experiment 1.5 convention, bound to 127.0.0.1 and started on the host whose port is forwarded, not on an unrelated compute node. Forward that port privately through VS Code; no public tunnel.

- Log train loss, learning rate, gradient norm, score variance, validation Spearman by task and market, throughput, GPU memory, and checkpoint duration. Flush and back up only approved aggregate event files.

  

## Backups

  

- GitHub: back up to a folder under https://github.com/YLiu95/MSc-new-experiments, for example `experiment_1.7_asc_gpu_nodes/`, following the 1.6 layout of source, tests, configs, environment lock, documentation, diary, and aggregate reports. Verify visibility, branch, and digests after the push; the repository currently reads as public, so nothing private may be placed there.

- Hugging Face: create a dedicated model repo, for example `YL95/experiment-1.7-asc-gpu-nodes`, and back up the best and latest checkpoints. If best and latest are the same checkpoint, back up only the best, as 1.6 does: the best inference export is uploaded and the full resumable latest stays in verified private storage.

- GITHUB_TOKEN and HF_TOKEN available at "/net/people/tutorial/tutorial042/.env"

- Keep raw price panels, per-sample labels and predictions, and intermediate checkpoints off the public repositories.

  

## Open gates


- Open gates before launch: confirm actual GPU access and hours; fix the concrete above-10B architecture and parallel layout; decide whether 1.7 keeps 1.6's nonrepetition sampler and reserved evaluation set; freeze the exact configuration before training starts.