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
	The hardware gate is not yet passed; production is not submitted.
- Replacement job `285957.hunter-pbs01` started at 21:12:07 CEST, eight seconds
	after submission, on the same two nodes. It retains all frozen input hashes.
	Its backward/Adam step took 11.0299 seconds with 39.578125 GiB peak reserved
	memory; replacement backup timing and final recovery results are pending.
- At 21:16 CEST only four MI300A nodes were free. Neither a two-node nor a
	four-node production request satisfies the strict below-20-percent rule at
	that availability; capacity must be checked again before submission.
- The Windows-initiated local SSH publication forward was verified. No new
	checkpoint or TensorBoard event has yet been verified on Hugging Face.
	The five-minute credential-local publication relay has been started.

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
and is being tested by the replacement job. The 600-second timing requirement
has not been relaxed.

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
