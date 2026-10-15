---
language: en
tags:
- pytorch
- safetensors
- tensorboard
- finance
- time-series
base_model: YL95/experiment-1.9-hunter-apu-nodes
---

# Experiment 1.9.1: Twelve-Node Hunter Ranker

This 1,069,302,417-parameter ranker continues Experiment 1.9 from its best
checkpoint, step 328 / 104,960 scheduled samples. The parent's monitoring
Spearman was 0.08534306347402562, not a new Experiment 1.9.1 test result.
The parent architecture and pinned data/monitoring split are unchanged.

The requested run uses twelve Hunter MI300A nodes, four GPU workers each,
BF16 compute and FP32 master weights/Adam. The per-update learning rate is
capped at 0.00006. Global batch increases from 320 to 1536, with microbatch 2
and balanced gradient accumulation. Adam and sampler state are restored;
RNG streams are reseeded for 48 workers. This is not bitwise continuation.

Best/latest inference exports, original TensorBoard events and aggregate run
reports are published here. A queued allocation or a source backup does not
prove the 48-GPU hardware gate or training has started. Consult the scheduler,
preflight, migration and completion reports for observed status.

Routine checkpoints follow five-minute monitoring validation. Final saving
targets completion two minutes before the original seven-hour PBS allocation
ends. Public inference exports cannot fully resume training; full Adam/RNG/
sampler checkpoints are backed up separately to a private recovery repository.
No private panels, credentials or full recovery payloads are published here.

[Source code](https://github.com/YLiu95/MSc-new-experiments/tree/main/experiment%201.9.1%20Hunter%20apu%20nodes).
[TensorBoard](https://huggingface.co/YL95/experiment-1.9.1-hunter-apu-nodes/tensorboard).

This is research software, not a validated trading system or financial advice.