# Experiment 1.7 work diary

2026-09-29: Implemented the requested experiment from the corrected 1.6 sampling/data lineage and 1.5 tensor-parallel primitives. CPU fake-tensor instantiation measured 10,193,843,857 parameters. Tiny-model forward/backward and all-configuration catalogue tests passed.

Decisions: retain nonrepetition; do not reserve the three named task configurations; select checkpoints using the frozen primary monitoring subset and explicitly label the reduced evaluation coverage. Interpret the plan's conflicting per-GPU replica phrase as eight TP shards per node, with one full logical model replica per node and DP between nodes.

Full GPU preflight, actual allocation, throughput, checkpoint/backup durations, and bitwise next-update equivalence are launch gates, not assumed results. Per-job preflight and final backup reports provide measured outcomes. A failed gate must not be represented as a training run.