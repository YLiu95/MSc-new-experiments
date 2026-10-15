# Hunter Port Status

## Final Results Checked on 2026-10-15

This section supersedes the pending-run status in the historical deployment
record below. Six-node job `286067.hunter-pbs01` started at 00:14:39 CEST and
ended unsuccessfully at 07:13:35 CEST, with PBS exit status 127 and reported
resource walltime 06:57:41. No `TRAINING_DONE.json` exists for this job.

The full 24-worker preflight passed: checkpoint save 148.342005148 seconds,
verified backup 178.938181509 seconds (327.280186657 combined), peak reserved
memory 46.076171875 GiB, bitwise next-update replay and sampler replay true.
Migration from step 192 succeeded, preserving FP32 weights, Adam, sampler and
scheduled count while reseeding RNG as approved. The new baseline and backup
were verified before the old parent recovery files were retired.

Training logged 218 completed updates, steps 193 through 410, adding 69,760
scheduled samples and reaching 82,240 since the Hunter warm start. The last
loss was 0.6902306942269206 and learning rate 0.000082. Update-only throughput
averaged 10.524276935052752 samples/second, excluding preflight, validation and
checkpoint overhead. Maximum reserved GPU memory was 59.623046875 GiB.

There were 33 monitoring-subset validation records. Best Spearman was
0.08394159463901718 at step 305; the last was 0.06689862950127207 at step 355.
No full-validation or held-out-test result is claimed. Best inference weights
are retained at step 305 (48,640 scheduled samples). Latest inference and full
recovery are step 355 (64,640 scheduled samples). The 55 updates and 17,600
scheduled samples after step 355 have logs but no complete saved checkpoint.

The first observed error was rank 4, then rank 8, failing to save RNG state
because `step_00000410_000000082240.staging` did not exist. The traceback enters
the timed checkpoint branch in `ranking17/train.py` and `ranking17/recovery.py`.
Independent rank-local timing decisions are a suspected coordination defect;
the root cause is not proven and no training-code fix or restart was performed.

The latest recovery and same-filesystem backup have identical 35-file manifests,
with 142,770,222,377 payload bytes and manifest SHA256
`2215625785a655c79e6669d391742005f24896295fcb0db286aa34c3d2fda57f`.
Their completion-marker hashes, file existence and sizes passed inspection,
as did all six best-export files. This inspection did not rehash the full
recovery payloads; the recorded full backup verification had passed previously.
Public backups exclude FP32 recovery/Adam, RNG, sampler and private data.
The Hunter recovery copies share a temporary filesystem and are not permanent
off-site recovery backups. Account expiry could not be determined from the
ordinary login account; workspace lifetime and PBS deadlines are separate.

## Historical Deployment Record

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

## Approved Six-Node Handoff

On 2026-10-14 the user requested six nodes until 08:00 CEST on 15 October and
explicitly waived the below-20-percent rule for this request. At 23:34 CEST,
22 MI300A nodes were free. The user separately approved non-bitwise topology
migration preserving FP32 weights, Adam, sampler progress and the step counter,
with reseeded per-rank RNG streams. Global batch 320 and TP4 are unchanged.

The migration source is isolated in `source-six-node-20261014`; the original
job's source was not overwritten. The initial handoff-controller archive SHA256 is
`4bdd8c9c9ee966d137574118d5dce1f0b2919a2a13e5922ad60b603ea6c75fc2`,
backed up in GitHub commit `7ac7697dd9d68282a808b0eabb3e380430b83256`.
Hunter passed all 44 tests in 4.59 seconds plus shell syntax checks. The most
recent focused Windows checks passed 27 tests with the POSIX-only case excluded.
An initial isolated test command accidentally replaced the module PYTHONPATH;
preserving the inherited paths corrected collection without installing packages.

A graceful STOP marker was sent to job `285983.hunter-pbs01`. Its final training
record confirms step 192 and 12,480 scheduled samples, with best monitoring
metric 0.0794105090388314. The final backup passed verification in 296.3562
seconds; checkpoint manifest SHA256 is
`61b85c551eeb758c2ab183304c2818a5eeadb490b39333409964c6d6069b3150`.
The detached one-time handoff controller was started on Hunter as PID 3635320.
It requires successful source-job completion and verifies both full copies
before removing only the redundant backup to make room for the new gate.
The original checkpoint remains until a migrated recovery/backup pair has
passed verification; the old best inference export and provenance are retained.
The 600-second queue limit, full 24-worker preflight, 20-minute shutdown reserve
and 08:00 CEST deadline remain enforced. Submission and hardware migration
results must be checked in the runtime handoff and job reports; preparation
alone is not evidence of successful six-node training.

Before submission, the handoff controller was paused during final checkpoint
verification to correct migration reseeding: the seed is shared by ranks in
the same TP4 group and differs across DP replicas, preserving replicated
dropout behavior. The focused migration tests passed with explicit same-TP
and different-DP assertions. No six-node job had been submitted at that point.

The corrected deployment archive is
`085df23f397510a264edb278577b396dbb8f4441d22ecf7fc9b3280c915d937a`,
backed up in GitHub commit `fe0ea023be7c23bd6c0d2de9f949cabd968ee34b`.
All 44 Hunter tests passed again in 4.69 seconds, and shell syntax passed.
The same handoff controller resumed and submitted job `286067.hunter-pbs01`
for six nodes; six MI300A nodes were free at submission. The explicit fraction
waiver was recorded. The original allocation finished successfully and both
full checkpoint copies were verified before the redundant backup was removed,
freeing 142,769,244,942 payload bytes while retaining the original checkpoint.

While the job queued, PBS walltime was reduced from 07:51:18 to 07:41:18 so
even a start near the ten-minute queue limit would respect the absolute end.
PBS initially reported insufficient Qlist resources. It subsequently started
at 00:14:39 CEST on 2026-10-15, about 6 minutes 57 seconds after submission.
At 00:17:48 PBS confirmed six nodes running. The user then explicitly chose
to keep those already allocated six nodes instead of replacing them with four
or two. No replacement reservation was submitted.

The verified allocation end is 07:55:57 CEST, with an application cutoff of
07:35:57 CEST and a 20-minute shutdown reserve. Frozen runtime source SHA256 is
`8c69e25bf91e8954e174d94d81f7a74cf45d3f3957ccb2f531614c215709b131`.
The run configuration binds the verified step-192 parent manifest, 12,480
scheduled samples and preserved optimizer/sampler state. The 24-worker gate,
actual migrated baseline and first new training update were still pending at
that check; allocation alone is not proof that migration has completed.
