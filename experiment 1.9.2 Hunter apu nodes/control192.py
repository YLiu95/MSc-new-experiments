import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
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
        raise ValueError("Expected the new twelve-node 24-hour allocation")
    if job["job_state"] != "R" or not job.get("stime") or not job.get("exec_host"):
        return None
    hosts = list(dict.fromkeys(part.split("/")[0] for part in job["exec_host"].split("+")))
    if len(hosts) != NODES or any(not re.fullmatch(r"x\d+c\d+s\d+b\d+n\d+", host) for host in hosts):
        raise ValueError("Invalid allocated Hunter host list")
    started = datetime.strptime(job["stime"], "%a %b %d %H:%M:%S %Y").replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    return {"hosts": hosts, "start_epoch": started, "end_epoch": started + 86400, "stop_epoch": started + 86280}


def launch_command(workspace, source, root, timing):
    environment = ["env", f"HUNTER_WORKSPACE={workspace}", f"PBS_JOBID={JOB}",
                   f"PBS_NODEFILE=/var/spool/pbs/aux/{JOB}", "PBS_MOMPORT=15003",
                   "PBS_ENVIRONMENT=PBS_INTERACTIVE", f"TRAIN_END_AT={timing['end_epoch']}"]
    command = [*environment, "/opt/pbs/bin/pbs_attach", "-j", JOB, "-s", "/usr/bin/bash", "-l", str(source / "run.sh")]
    control = root / "control"
    return ("mkdir " + shlex.quote(str(control / "start.lock")) + " || exit 0; nohup " + shlex.join(command)
            + " >" + shlex.quote(str(control / "launch.log")) + " 2>&1 </dev/null & "
            + "printf '%s\\n' \"$!\" >" + shlex.quote(str(control / "launcher.pid")))


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
        result[arm] = {"preflight_passed": (directory / "reports/preflight.json").is_file(),
                       "new_updates": record.get("new_updates", 0) if record else 0,
                       "last_update_time": record.get("time") if record else None,
                       "done": (directory / "TRAINING_DONE.json").is_file()}
    return result


def watch(workspace, source):
    root = workspace / "experiment192/artifacts" / f"job-{JOB}"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "watch.lock").mkdir()
    ready = json.loads((source / "READY.json").read_text())
    if not ready["tests_passed"] or any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in ready["files"].items()):
        raise ValueError("Deployment differs from validated source")
    if not (workspace / "experiment192/prepared/COMPLETE").is_file():
        raise ValueError("Private cohort preparation has not completed")
    previous, heartbeat, failures, notified = None, 0, 0, set()
    while True:
        try:
            response = subprocess.run(["/opt/pbs/bin/qstat", "-xf", "-F", "json", JOB],
                                      check=True, capture_output=True, text=True, timeout=30)
            job = json.loads(response.stdout)["Jobs"][JOB]
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
        if timing is not None and not (control / "start.lock").exists():
            if now + 900 >= timing["stop_epoch"]:
                raise RuntimeError("Insufficient allocation time for safe startup")
            write_json(control / "deadlines.json", {"job": JOB, **timing})
            try:
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", timing["hosts"][0],
                                         launch_command(workspace, source, root, timing)],
                                        env={**os.environ, "PBS_JOBID": JOB}, capture_output=True, text=True, timeout=60)
                if result.returncode:
                    print(json.dumps({"event": "job_bound_launch_retry", "error": result.stderr.strip(), "time": now}), flush=True)
            except (OSError, subprocess.SubprocessError) as error:
                print(json.dumps({"event": "job_bound_launch_retry", "error": str(error), "time": now}), flush=True)
            if (control / "start.lock").exists():
                write_json(control / "launch.json", {"event": "launch_requested", "job": JOB, "time": now, **timing})
                print(json.dumps({"event": "ALLOCATION_READY_LAUNCH_REQUESTED", "job": JOB, **timing}), flush=True)
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
            write_json(control / "allocation-ended.json", {"job": JOB, "arms": current, "time": now})
            print(json.dumps({"event": "allocation_ended", "job": JOB, "arms": current}), flush=True)
            return
        threading.Event().wait(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    watch(arguments.workspace, arguments.source)