import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

from policy192 import JOB, NODES


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def allocation(job):
    resources = job["Resource_List"]
    if int(resources["nodect"]) != NODES or resources["walltime"] != "24:00:00":
        raise ValueError(f"Expected a {NODES}-node 24-hour allocation")
    if job["job_state"] != "R" or not job.get("stime") or not job.get("exec_host"):
        return None
    hosts = list(dict.fromkeys(part.split("/")[0] for part in job["exec_host"].split("+")))
    if len(hosts) != NODES or any(not re.fullmatch(r"x\d+c\d+s\d+b\d+n\d+", host) for host in hosts):
        raise ValueError("Invalid allocated Hunter host list")
    started = datetime.strptime(job["stime"], "%a %b %d %H:%M:%S %Y").replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    return {"hosts": hosts, "start_epoch": started, "end_epoch": started + 86400, "stop_epoch": started + 86280}


def validate_source(workspace, source):
    ready = json.loads((source / "READY.json").read_text())
    if not ready["tests_passed"] or any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in ready["files"].items()):
        raise ValueError("Deployment differs from validated source")
    if not (workspace / "experiment192/prepared/COMPLETE").is_file():
        raise ValueError("Private cohort preparation has not completed")


def scheduler_job():
    if not re.fullmatch(r"\d+\.hunter-pbs01", JOB):
        raise ValueError("EXPERIMENT192_JOB must identify the submitted Hunter job")
    response = subprocess.run(["/opt/pbs/bin/qstat", "-xf", "-F", "json", JOB],
                              check=True, capture_output=True, text=True, timeout=30)
    return json.loads(response.stdout)["Jobs"][JOB]


def batch_start(workspace, source):
    if os.environ.get("PBS_JOBID") != JOB or os.environ.get("PBS_ENVIRONMENT") != "PBS_BATCH":
        raise ValueError("Startup must run inside the native PBS batch job")
    validate_source(workspace, source)
    timing = allocation(scheduler_job())
    if timing is None or time.time() + 900 >= timing["stop_epoch"]:
        raise ValueError("Batch allocation is not ready or lacks startup reserve")
    control = workspace / "experiment192/artifacts" / f"job-{JOB}" / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "start.lock").mkdir()
    write_json(control / "deadlines.json", {"job": JOB, **timing})
    write_json(control / "launch.json", {"event": "native_batch_started", "job": JOB,
                                        "source": str(source), "time": time.time(), **timing})
    return timing["end_epoch"]


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
        result[arm] = {"preflight_passed": (directory / "reports/preflight.json").is_file(),
                       "initial_checkpoint_complete": complete,
                       "new_updates": record.get("new_updates", 0) if record else 0,
                       "last_update_time": record.get("time") if record else None,
                       "done": (directory / "TRAINING_DONE.json").is_file()}
    return result


def watch(workspace, source):
    root = workspace / "experiment192/artifacts" / f"job-{JOB}"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "watch.lock").mkdir()
    validate_source(workspace, source)
    previous, heartbeat, failures, notified = None, 0, 0, set()
    while True:
        try:
            job = scheduler_job()
            allocation(job)
            failures = 0
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            failures += 1
            record = {"event": "scheduler_retry", "job": JOB, "attempt": failures, "error": str(error), "time": time.time()}
            write_json(control / "status.json", record)
            print(json.dumps(record), flush=True)
            threading.Event().wait(min(30, 5 * failures))
            continue
        now = time.time()
        current = progress(root)
        fingerprint = (job["job_state"], tuple((arm, values["preflight_passed"], values["new_updates"], values["done"]) for arm, values in current.items()))
        if fingerprint != previous or now - heartbeat >= 60:
            record = {"event": "monitor", "job": JOB, "state": job["job_state"], "comment": job.get("comment"),
                      "launch_requested": (control / "start.lock").exists(), "arms": current, "time": now}
            write_json(control / "status.json", record)
            print(json.dumps(record), flush=True)
            previous, heartbeat = fingerprint, now
        if all(values["new_updates"] > 0 for values in current.values()) and "training" not in notified:
            print(json.dumps({"event": "BOTH_ARMS_TRAINING", "job": JOB, "arms": current, "time": now}), flush=True)
            notified.add("training")
        launch_path = control / "launch.json"
        if launch_path.is_file():
            launch_at = json.loads(launch_path.read_text())["time"]
            for arm, values in current.items():
                reference = values["last_update_time"] or launch_at
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
    arguments = parser.parse_args()
    if arguments.batch_start:
        print(batch_start(arguments.workspace, arguments.source))
    elif arguments.batch_exit is not None:
        batch_exit(arguments.workspace, arguments.batch_exit)
    else:
        watch(arguments.workspace, arguments.source)