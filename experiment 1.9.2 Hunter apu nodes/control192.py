import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

from policy192 import JOB, NODES, WALLTIME, WALLTIME_SECONDS
from storage192 import require_headroom


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def allocation(job):
    resources = job["Resource_List"]
    if int(resources["nodect"]) != NODES or resources["walltime"] != WALLTIME:
        raise ValueError(f"Expected a {NODES}-node {WALLTIME} allocation")
    if int(resources.get("ncpus", 0)) != NODES * 96 or resources.get("place") != "scatter:excl":
        raise ValueError("Expected exclusive nodes with 96 CPU cores each")
    if job["job_state"] != "R" or not job.get("stime") or not job.get("exec_host"):
        return None
    hosts = list(dict.fromkeys(part.split("/")[0] for part in job["exec_host"].split("+")))
    if len(hosts) != NODES or any(not re.fullmatch(r"x\d+c\d+s\d+b\d+n\d+", host) for host in hosts):
        raise ValueError("Invalid allocated Hunter host list")
    started = datetime.strptime(job["stime"], "%a %b %d %H:%M:%S %Y").replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    return {"hosts": hosts, "start_epoch": started, "end_epoch": started + WALLTIME_SECONDS,
            "stop_epoch": started + WALLTIME_SECONDS - 120}


def validate_source(workspace, source):
    ready = json.loads((source / "READY.json").read_text())
    if not ready["tests_passed"] or any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in ready["files"].items()):
        raise ValueError("Deployment differs from validated source")
    if not (workspace / "experiment192/prepared/COMPLETE").is_file():
        raise ValueError("Private cohort preparation has not completed")


def scheduler_job(job_id=None):
    job_id = job_id or JOB
    if not re.fullmatch(r"\d+\.hunter-pbs01", job_id):
        raise ValueError("EXPERIMENT192_JOB must identify the submitted Hunter job")
    response = subprocess.run(["/opt/pbs/bin/qstat", "-xf", "-F", "json", job_id],
                              check=True, capture_output=True, text=True, timeout=30)
    return json.loads(response.stdout)["Jobs"][job_id]


def prepare_continuation(workspace, source, root):
    import fcntl
    from evaluate192 import selection_complete
    base = workspace / "experiment192"
    journal = base / "continuation-finish-b"
    journal.mkdir(exist_ok=True)
    with (journal / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = base / "artifacts/job-287999.hunter-pbs01"
        if (journal / "latest.json").exists():
            record = json.loads((journal / "latest.json").read_text())
            if scheduler_job(record["job"])["job_state"] != "F":
                raise ValueError("Previous continuation job is still active")
            previous = Path(record["root"]).resolve()
            previous.relative_to((base / "artifacts").resolve())
        resume = previous / "arm-B"
        if not (resume / "latest.json").exists():
            resume = Path(json.loads((previous / "control/continuation.json").read_text())["resume_arm_root"])
        resume.resolve().relative_to((base / "artifacts").resolve())
        if not (resume / "latest.json").exists():
            raise ValueError("No complete Arm B resume pointer")
        original_a = base / "artifacts/job-287999.hunter-pbs01/arm-A"
        if not selection_complete(json.loads((original_a / "TRAINING_DONE.json").read_text())):
            raise ValueError("Original Arm A selection is not frozen")
        (root / "arm-A").symlink_to(original_a, target_is_directory=True)
        done = resume / "TRAINING_DONE.json"
        completed = done.is_file() and selection_complete(json.loads(done.read_text()))
        if (previous / "final-test/OPENED.json").exists() and not completed:
            raise ValueError("Opened final test cannot resume unfrozen training")
        if completed:
            def copy_payload(incoming, outgoing):
                if "checkpoints" in Path(incoming).parts or "exports" in Path(incoming).parts:
                    return os.link(incoming, outgoing)
                return shutil.copy2(incoming, outgoing)
            shutil.copytree(resume, root / "arm-B", copy_function=copy_payload,
                            ignore=shutil.ignore_patterns("control", "logs", "ledgers"))
        if (previous / "final-test").exists():
            shutil.copytree(previous / "final-test", root / "final-test")
        write_json(root / "control/continuation.json", {"resume_arm_root": str(resume),
                   "arm_a_root": str(original_a), "training_already_complete": bool(completed),
                   "previous_root": str(previous), "job": JOB, "source": str(source)})
        write_json(journal / "latest.json", {"job": JOB, "root": str(root), "source": str(source)})


def batch_start(workspace, source):
    if os.environ.get("PBS_JOBID") != JOB or os.environ.get("PBS_ENVIRONMENT") != "PBS_BATCH":
        raise ValueError("Startup must run inside the native PBS batch job")
    validate_source(workspace, source)
    timing = allocation(scheduler_job())
    if timing is None or time.time() + 900 >= timing["stop_epoch"]:
        raise ValueError("Batch allocation is not ready or lacks startup reserve")
    validate_compute(timing)
    storage = require_headroom(workspace)
    control = workspace / "experiment192/artifacts" / f"job-{JOB}" / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "start.lock").mkdir()
    prepare_continuation(workspace, source, control.parent)
    write_json(control / "storage-admission.json", storage)
    write_json(control / "deadlines.json", {"job": JOB, **timing})
    write_json(control / "launch.json", {"event": "native_batch_started", "job": JOB,
                                        "source": str(source), "time": time.time(), **timing})
    return timing["end_epoch"]


def validate_compute(timing):
    nodefile = Path(f"/var/spool/pbs/aux/{JOB}")
    if socket.gethostname().split(".")[0] != timing["hosts"][0]:
        raise ValueError("Dispatch must execute on this allocation's master node")
    hosts = list(dict.fromkeys(line.split(".")[0] for line in nodefile.read_text().splitlines()))
    if hosts != timing["hosts"]:
        raise ValueError("PBS node file differs from the live allocation")
    return nodefile


def dispatch(workspace, source):
    validate_source(workspace, source)
    timing = allocation(scheduler_job())
    if timing is None or time.time() + 900 >= timing["stop_epoch"]:
        raise ValueError("Interactive allocation lacks startup reserve")
    nodefile = validate_compute(timing)
    require_headroom(workspace)
    control = workspace / "experiment192/artifacts" / f"job-{JOB}" / "control"
    control.mkdir(parents=True, exist_ok=True)
    try:
        (control / "dispatch.lock").mkdir()
    except FileExistsError:
        return
    environment = {**os.environ, "PBS_JOBID": JOB, "PBS_ENVIRONMENT": "PBS_INTERACTIVE",
                   "PBS_NODEFILE": str(nodefile), "PBS_MOMPORT": "15003", "PBS_NODENUM": "0", "PBS_TASKNUM": "1",
                   "TRAIN192_ATTACHED": JOB, "TRAIN192_SOURCE": str(source), "HUNTER_WORKSPACE": str(workspace)}
    with (control / "launch.log").open("ab") as stream:
        child = subprocess.Popen(["/opt/pbs/bin/pbs_attach", "-j", JOB, "-s", "/usr/bin/bash", str(source / "run.sh")],
                                 env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    write_json(control / "dispatch.json", {"job": JOB, "pid": child.pid, "host": timing["hosts"][0], "time": time.time()})


def request_launch(workspace, source, timing):
    shell = (f"export HUNTER_WORKSPACE={shlex.quote(str(workspace))} EXPERIMENT192_JOB={shlex.quote(JOB)}; "
             f"source {shlex.quote(str(workspace / 'source-six-node-20261014/env.sh'))}; "
             f"export PYTHONPATH={shlex.quote(str(source))}:\"$PYTHONPATH\"; "
             f"export RESUME192_ROOT={shlex.quote(os.environ['RESUME192_ROOT'])}; "
             + shlex.join([str(workspace / "venv/bin/python"), str(source / "control192.py"),
                           "--workspace", str(workspace), "--source", str(source), "--dispatch"]))
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", timing["hosts"][0],
                    "bash -lc " + shlex.quote(shell)], env={**os.environ, "PBS_JOBID": JOB},
                   check=True, capture_output=True, text=True, timeout=60)


def batch_exit(workspace, exit_code):
    control = workspace / "experiment192/artifacts" / f"job-{JOB}" / "control"
    record = {"event": "batch_exit", "job": JOB, "exit_code": exit_code, "time": time.time()}
    write_json(control / "batch-exit.json", record)
    if exit_code:
        write_json(control / "failure.json", record)


def progress(root):
    result = {}
    for arm in ("A", "B"):
        directory = root / f"arm-{arm}"
        history, record = directory / "history.jsonl", None
        if history.is_file():
            with history.open("rb") as stream:
                stream.seek(max(0, history.stat().st_size - 32768))
                for line in reversed(stream.read().splitlines()):
                    try:
                        record = json.loads(line)
                        break
                    except ValueError:
                        continue
        complete = False
        pointer = directory / "latest.json"
        if pointer.is_file():
            try:
                checkpoint = (directory / json.loads(pointer.read_text())["directory"]).resolve()
                checkpoint.relative_to(directory.resolve())
                complete = (checkpoint / "COMPLETE").is_file()
            except (OSError, ValueError, KeyError):
                pass
        preflight_path = directory / "reports/preflight.json"
        preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else {}
        updates = record.get("new_updates", 0) if record else 0
        starting = preflight.get("starting_new_updates", updates)
        result[arm] = {"preflight_passed": preflight.get("passed") is True,
                       "initial_checkpoint_complete": complete,
                   "new_updates": updates, "starting_new_updates": starting,
                   "fresh_updates": max(0, updates - starting),
                       "last_update_time": record.get("time") if record else None,
                       "done": (directory / "TRAINING_DONE.json").is_file()}
    return result


def watch(workspace, source, autolaunch=False):
    root = workspace / "experiment192/artifacts" / f"job-{JOB}"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "watch.lock").mkdir()
    validate_source(workspace, source)
    previous, heartbeat, failures, notified, launch_attempts = None, 0, 0, set(), 0
    while True:
        try:
            job = scheduler_job()
            timing = allocation(job)
            failures = 0
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            failures += 1
            record = {"event": "scheduler_retry", "job": JOB, "attempt": failures, "error": str(error), "time": time.time()}
            write_json(control / "status.json", record)
            print(json.dumps(record), flush=True)
            threading.Event().wait(min(30, 5 * failures))
            continue
        now = time.time()
        if autolaunch and timing and not (control / "dispatch.lock").exists() and launch_attempts < 12:
            launch_attempts += 1
            try:
                request_launch(workspace, source, timing)
            except (OSError, subprocess.SubprocessError) as error:
                write_json(control / "launch-retry.json", {"attempt": launch_attempts, "error": str(error), "time": time.time()})
                if launch_attempts == 12:
                    write_json(control / "failure.json", {"event": "launch_failed", "error": str(error), "time": time.time()})
        if (control / "dispatch.json").is_file() and not (control / "launch.json").is_file():
            dispatched = json.loads((control / "dispatch.json").read_text())
            if now - dispatched["time"] > 180:
                write_json(control / "failure.json", {"event": "attachment_failed", "dispatch": dispatched, "time": now})
        current = progress(root)
        fingerprint = (job["job_state"], tuple((arm, values["preflight_passed"], values["new_updates"], values["done"]) for arm, values in current.items()))
        if fingerprint != previous or now - heartbeat >= 60:
            record = {"event": "monitor", "job": JOB, "state": job["job_state"], "comment": job.get("comment"),
                      "launch_requested": (control / "start.lock").exists(), "arms": current, "time": now}
            write_json(control / "status.json", record)
            print(json.dumps(record), flush=True)
            previous, heartbeat = fingerprint, now
        if all(values["preflight_passed"] and values["fresh_updates"] > 0 for values in current.values()) and "training" not in notified:
            print(json.dumps({"event": "BOTH_ARMS_TRAINING", "job": JOB, "arms": current, "time": now}), flush=True)
            notified.add("training")
        launch_path = control / "launch.json"
        if launch_path.is_file():
            launch_at = json.loads(launch_path.read_text())["time"]
            for arm, values in current.items():
                reference = max(values["last_update_time"] or launch_at, launch_at)
                if not values["done"] and now - reference > 1200 and arm not in notified:
                    alert = {"event": "INTERVENTION_REQUIRED", "job": JOB, "arm": arm,
                             "reason": "No completed update in twenty minutes", "time": now}
                    write_json(control / "failure.json", alert)
                    print(json.dumps(alert), flush=True)
                    notified.add(arm)
        if (root / "EXPERIMENT_DONE.json").exists():
            print(json.dumps({"event": "experiment_complete", "job": JOB}), flush=True)
            return
        if job["job_state"] == "F":
            write_json(control / "allocation-ended.json", {"job": JOB, "arms": current, "time": now,
                                                          "exit_status": job.get("Exit_status"), "comment": job.get("comment")})
            print(json.dumps({"event": "allocation_ended", "job": JOB, "arms": current}), flush=True)
            return
        threading.Event().wait(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--batch-start", action="store_true")
    action.add_argument("--batch-exit", type=int)
    action.add_argument("--resume-root", action="store_true")
    action.add_argument("--dispatch", action="store_true")
    action.add_argument("--autolaunch", action="store_true")
    arguments = parser.parse_args()
    if arguments.batch_start:
        print(batch_start(arguments.workspace, arguments.source))
    elif arguments.resume_root:
        record = arguments.workspace / "experiment192/artifacts" / f"job-{JOB}" / "control/continuation.json"
        print(json.loads(record.read_text())["resume_arm_root"])
    elif arguments.batch_exit is not None:
        batch_exit(arguments.workspace, arguments.batch_exit)
    elif arguments.dispatch:
        dispatch(arguments.workspace, arguments.source)
    else:
        watch(arguments.workspace, arguments.source, arguments.autolaunch)