# Experiment 1.7.1 Execution Plan

1. Verify pinned source, private dataset and step-153 inference files.
2. Test component-aware TP8-to-TP4 conversion and PBS deadline/resource policy.
3. Stage all inputs in Lustre; preserve the HLRS PyTorch/RCCL module.
4. Request two nodes for one hour, cancelling after ten minutes queued.
5. Reconstruct and hash-check the original vocabulary/validation manifest.
6. Run full-model backward, fused AdamW, checkpoint, backup, bitwise next-update
	restore and sampler continuity tests; record per-rank CPU/HIP memory.
7. On success, request 2 or 4 nodes strictly below 20% of free APUs, preferring 4.
8. Repeat the full gate at production topology before any training update.
9. Preserve dropout, global batch 320, microbatch 1, and the learning-rate schedule.
10. Save locally and publish allowlisted inference/events through a separate relay.
11. Stop before 2026-10-15 08:00 CEST, reserving at least 20 minutes for shutdown.

This is an approved warm start with Adam/RNG/sampler reset, not exact ASC recovery.
Do not enable speculative batching, mask labels in attention, relax the restore
gate silently, or start automatic afterany job chains.