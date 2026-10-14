# Hunter Port Status

As of initial source backup on 2026-10-14:

- Hunter access and a 30-day Lustre workspace are established.
- Actual runtime is PyTorch 2.10.0.dev20251031+rocm6.4 with HIP 6.4.4.
- The module loads the AWS OFI RCCL plugin; it is not replaced.
- Dataset and step-153 inference downloads passed size/digest verification.
- PBS policy, tensor conversion, model/data contract, retention and publication
	checks pass locally. POSIX permissions require the Linux test run.
- The full model has not yet passed the Hunter hardware preflight at this stage.
- No production-training results are claimed by this initial report.

Runtime reports under each job's `reports/` are the source of execution evidence.
The user approved resetting Adam/RNG/sampler because supplied public repositories
contain inference files only. Private training panels and recovery states are
never included in public source or inference uploads.
