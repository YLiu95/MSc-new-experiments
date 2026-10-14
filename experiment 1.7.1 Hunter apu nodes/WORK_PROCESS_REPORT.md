# Hunter Port Status

Verified deployment status on 2026-10-14:

- Hunter access and a 30-day Lustre workspace are established.
- Actual runtime is PyTorch 2.10.0.dev20251031+rocm6.4 with HIP 6.4.4.
- The module loads the AWS OFI RCCL plugin; it is not replaced.
- All 142 staged files (94 dataset files, 38 wheels and 10 warm-start files;
	25,634,971,625 bytes) passed size/SHA256 verification on Hunter. Interrupted
	transfers were resumed using client-initiated SFTP and a fresh size inventory.
- The revised Hunter Linux suite passed all 32 tests in 4.65 seconds, including
	POSIX permissions. Windows passed 31 tests with that POSIX-only test deselected.
- All 13 markets were prepared offline. Vocabulary SHA256 is
	`7e713177fd22930fc3c761ff045b0e020e35526496d3a37e3b087a40cec84c29`;
	primary validation SHA256 is
	`fe47c9d6aeecf6a84ad1ec25504ab77471767f08def34ce0c8965f7783cbcd50`.
	Both match the parent experiment. There are 51,712 primary baskets, 1,616
	monitoring baskets and 19,456 held-out baskets; no test evaluation is claimed.
- Two-node, one-hour preflight job `285926.hunter-pbs01` started at
	20:42:09 CEST, eight seconds after submission. Its queue watchdog confirmed
	the running state before the 600-second cancellation deadline.
- All eight workers initialized TP4/DP2 on MI300A and passed the inter-node
	RCCL reduction. The worst-case backward/Adam step took 11.1127 seconds;
	maximum reserved GPU memory across ranks was 39.578125 GiB.
- Checkpoint save and verification took 146.8663 seconds. The backup passed
	verification in 536.4795 seconds, but their combined 683.3458 seconds exceeds
	the required 600-second shutdown-budget gate. Execution reached this final
	timing check after the bitwise restore-next-update and sampler comparisons,
	then raised the shutdown-reserve error. PBS reported finished/failed, exit
	status 127, after 25 minutes 57 seconds. The synthetic checkpoint copies were
	removed after preserving their matching manifests, configuration and state;
	dataset and warm-start files were retained.
	That first job did not pass the hardware gate.
- Replacement job `285957.hunter-pbs01` started at 21:12:07 CEST, eight seconds
	after submission, on the same two nodes. It retains all frozen input hashes.
	Its backward/Adam step took 11.0299 seconds with 39.578125 GiB peak reserved
	memory. Save took 146.8729 seconds and verified backup took 185.0916 seconds:
	331.9645 seconds combined, below the 600-second limit. Bitwise next-update
	replay and sampler replay passed on all eight workers as applicable. PBS
	reported successful completion with exit status zero; the gate marker exists.
- At 21:16 CEST only four MI300A nodes were free. Neither a two-node nor a
	four-node production request satisfies the strict below-20-percent rule at
	that availability. Capacity improved before the official submission below.
- Official job `285983.hunter-pbs01` requested two nodes with 12 free at
	submission (16.67 percent). It started at 21:29:25 CEST after eight seconds
	queued, with walltime 10:29:44. Its running PBS allocation ends on
	2026-10-15 at 07:59:09 CEST; the application cutoff is 07:39:09 CEST,
	leaving 20 minutes for checkpointing before the absolute 08:00 deadline.
	The official job repeats the matching hardware gate before training updates.
- The Windows-initiated local SSH publication forward was verified. Hugging
	Face commit `58b897f5d41b71eddfff61c41e956b566ff976e6` contains the model
	card, passed-preflight report, warm-start provenance and TensorBoard events;
	all four files passed remote size/digest verification. The hosted Training
	metrics tab exists. These are preflight metrics, not trained-model results.
	The five-minute credential-local relay is running for subsequent artifacts.

The first compute deployment is archived in GitHub commit
`7f01ca191d2cd6bb6f99b54de556427dc9fa28ac`, with frozen source SHA256
`0619bcc48ad49a14afa36cea84775744c561ca99b398d7a933bb0353f8a3e521`.
A subsequent Windows-only publication-cache fix removes outdated cache files
before replacement and requires 2 GiB of local free-space headroom. Its five
focused tests pass. It was kept separate from the first running compute job.

The measured backup timing prompted a separate recovery change: four workers
copy independent files and verify source/destination shard hashes concurrently.
Default verification elsewhere remains serial; every file is still checked,
and backup promotion remains atomic. Nine focused recovery/publication tests
passed (one POSIX-only test deselected), including concurrency and corrupted-copy
rejection. The revised source is deployed from GitHub commit
`bbdcf3cee3ae1dd6e520a8e4c80491416c284444`, with frozen source SHA256
`5910300725cd21a7b9a854f469419439be78ab7269a86de95a6a411a0d8319d0`,
and passed the replacement job. The 600-second timing requirement has not
been relaxed. The initial model-card publication encountered invalid tab
indentation in YAML; indentless tag sequences fixed it, and an additional
Hugging Face model-card parsing regression test passed locally. This metadata
correction does not change the deployed training code or its source fingerprint.

Runtime reports under each job's `reports/` are the source of execution evidence.
The user approved resetting Adam/RNG/sampler because supplied public repositories
contain inference files only. Private training panels and recovery states are
never included in public source or inference uploads.

The workspace environment was installed without replacing PyTorch. Pip reported
an unrelated inherited elasticsearch7/urllib3 constraint conflict; the experiment
suite passed, but this environment is not certified for elasticsearch7.

Two earlier transfer interruptions coincided with FortiClient VPN disconnects.
Reviewed client logs did not establish a security-policy trigger. Direct HTTPS
downloads from Hunter timed out; the completed transfer used client-initiated
SFTP, not reverse forwarding. The final resumed transfer ran at about 4 MB/s;
its success does not establish the cause of the earlier disconnects.
