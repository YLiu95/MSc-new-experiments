import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time

from control192 import allocation
from policy192 import JOB


def recover(workspace, run_payload, receiver_payload):
    import ast
    import base64
    import hashlib
    import signal

    base = workspace / "experiment192"
    control = base / "artifacts" / f"job-{JOB}" / "control"
    original = base / "source-287925"
    source = base / "source-287925-transport"
    pid = int((control / "watcher.pid").read_text())
    process_path = Path(f"/proc/{pid}")
    command = process_path.joinpath("cmdline").read_bytes().split(b"\0")
    if str(original / "control192.py").encode() not in command:
        raise ValueError("Recovery requires the original identified watcher")
    process_state = process_path.joinpath("status").read_text()
    if "State:\tT" not in process_state or (control / "start.lock").exists():
        raise ValueError("Recovery requires a stopped, never-launched watcher")
    run_bytes = base64.b64decode(run_payload, validate=True)
    receiver_bytes = base64.b64decode(receiver_payload, validate=True)
    ast.parse(receiver_bytes)
    ready = json.loads((original / "READY.json").read_text())
    if any(hashlib.sha256((original / name).read_bytes()).hexdigest() != checksum for name, checksum in ready["files"].items()):
        raise ValueError("Original source changed before recovery")
    shutil.copytree(original, source)
    (source / "run.sh").write_bytes(run_bytes)
    subprocess.run(["bash", "-n", str(source / "run.sh")], check=True)
    shells = run_bytes.decode().split("mpiexec -n 12")[1:]
    if len(shells) != 2 or any(not shell.index("source ") < shell.index("export NCCL_NET=Socket") < shell.index("torch.distributed.run") for shell in shells):
        raise ValueError("Missing post-module transport guard")
    environment = {**os.environ, "PYTHONPATH": str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([str(workspace / "venv/bin/python"), "-m", "pytest", str(source / "tests"), "-q"], env=environment, check=True)
    ready["files"]["run.sh"] = hashlib.sha256(run_bytes).hexdigest()
    ready["transport_shell_check"] = True
    (source / "READY.json").write_text(json.dumps(ready))
    (base / "receive192-v2.py").write_bytes(receiver_bytes)
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for attempt in range(50):
        if not process_path.exists() or "State:\tZ" in process_path.joinpath("status").read_text():
            break
        threading.Event().wait(0.1)
    else:
        raise RuntimeError("Original watcher has not exited; no duplicate started")
    (control / "watch.lock").rename(control / f"watch.lock.retired-{pid}")
    with (control / "watch.log").open("ab") as stream:
        watcher = subprocess.Popen([str(workspace / "venv/bin/python"), "-u", str(source / "control192.py"),
                                    "--workspace", str(workspace), "--source", str(source)], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (control / "watcher.pid").write_text(str(watcher.pid))
    (control / "transport-handover.json").write_text(json.dumps({"source": str(source), "watcher_pid": watcher.pid, "previous_pid": pid, "time": time.time()}))
    with (base / "receive192-v2.log").open("ab") as stream:
        receiver = subprocess.Popen([str(workspace / "venv/bin/python"), "-u", str(base / "receive192-v2.py"),
                                     "--sha256", "8eb61032e89b6ab53e925b0446d7ecace581b6310522b4c8672a00cd6ed805a4",
                                     "--bytes", "40960", "--destination", "source-287925-v2"], stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"event": "AUTOMATIC_LAUNCH_REARMED", "watcher_pid": watcher.pid, "receiver_pid": receiver.pid}), flush=True)


def reference(workspace):
    from torch.utils.tensorboard import SummaryWriter
    from ranker.data import write_json
    base = workspace / "experiment192"
    root = base / "artifacts" / f"job-{JOB}"
    previous = workspace / "experiment191/artifacts/job-287139.hunter-pbs01"
    destination = root / "runs/reference-1.9.1-ids-on"
    if destination.exists():
        raise ValueError("Reference curve already prepared")
    with SummaryWriter(str(destination)) as writer:
        for line in (previous / "history.jsonl").read_text().splitlines():
            row = json.loads(line)
            step = row["step"] - 328
            if not 0 <= step <= 1000:
                continue
            for name in ("loss", "pairs", "lr", "gradient_norm", "score_variance"):
                writer.add_scalar("train/" + name, row[name], step)
            writer.add_scalar("system/samples_per_second", row["samples_per_second"], step)
        for line in (previous / "validation.jsonl").read_text().splitlines():
            row = json.loads(line)
            step = row["step"] - 328
            if 0 <= step <= 1000:
                writer.add_scalar("validation/macro_spearman", row["macro_spearman"], step)
        writer.add_scalar("validation/macro_spearman", 0.08534306347402562, 0)
    for source in (previous / "runs").rglob("events.out.tfevents.*"):
        target = base / "reference-original" / source.relative_to(previous / "runs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(root / "reference.json", {"source": "Experiment 1.9.1 completed step-5411 run", "original_events_preserved": True,
               "derived_run": "reference-1.9.1-ids-on", "step_transform": "original step minus 328", "maximum_new_updates": 1000,
               "comparison_limitation": "Historical IDs-on reference, not a controlled arm differing only in identity masking."})
    print("REFERENCE_CURVE_READY", flush=True)


def audits(workspace, source):
    root = workspace / "experiment192/artifacts" / f"job-{JOB}"
    output = root / "audits"
    output.mkdir(parents=True, exist_ok=True)
    while True:
        response = subprocess.run(["/opt/pbs/bin/qstat", "-xf", "-F", "json", JOB], capture_output=True, text=True, timeout=30)
        if response.returncode:
            threading.Event().wait(30)
            continue
        job = json.loads(response.stdout)["Jobs"][JOB]
        if job["job_state"] == "F":
            return
        timing = allocation(job)
        if timing is None:
            threading.Event().wait(5)
            continue
        for action in ("universe", "exposure"):
            lock = output / f"{action}.lock"
            command = [str(workspace / "venv/bin/python"), "-u", str(source / "audit192.py"), action,
                       "--workspace", str(workspace), "--panel", str(workspace / "data/panel"), "--output", str(output)]
            environment = f"source {shlex.quote(str(workspace / 'source-six-node-20261014/env.sh'))}; export PYTHONPATH={shlex.quote(str(source))}:\"$PYTHONPATH\"; "
            shell = ("mkdir " + shlex.quote(str(lock)) + " || exit 0; " + environment + "nohup /opt/pbs/bin/pbs_attach -j "
                     + JOB + " -s /usr/bin/nice -n 10 /usr/bin/taskset -c 88-95 " + shlex.join(command)
                     + " >" + shlex.quote(str(output / f"{action}.log")) + " 2>&1 </dev/null &")
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", timing["hosts"][-1],
                                     "bash -l -c " + shlex.quote(shell)], env={**os.environ, "PBS_JOBID": JOB},
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError("Job-bound CPU audit launch failed: " + result.stderr)
        print("CPU_AUDITS_LAUNCHED", flush=True)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reference", "audits"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "reference":
        reference(arguments.workspace)
    else:
        audits(arguments.workspace, arguments.source)