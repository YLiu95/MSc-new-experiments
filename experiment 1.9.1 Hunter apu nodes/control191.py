import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

from ranker.data import write_json
from ranking17.recovery import digest
from train191 import verify_parent


JOB = "286972.hunter-pbs01"


def allocation(job):
    resources = job["Resource_List"]
    if int(resources["nodect"]) != 12 or resources["walltime"] != "07:00:00":
        raise ValueError("The requested twelve-node, seven-hour allocation changed")
    if job["job_state"] != "R":
        return None
    hosts = list(dict.fromkeys(part.split("/")[0] for part in job["exec_host"].split("+")))
    if len(hosts) != 12 or any(not re.fullmatch(r"x\d+c\d+s\d+b\d+n\d+", host) for host in hosts):
        raise ValueError("Expected twelve allocated Hunter compute hosts")
    started = datetime.strptime(job["stime"], "%a %b %d %H:%M:%S %Y").replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
    return {"hosts": hosts, "start_epoch": started, "end_epoch": started + 7 * 3600,
            "stop_epoch": started + 7 * 3600 - 120}


def launch_command(workspace, source, root, timing):
    environment = ["env", f"HUNTER_WORKSPACE={workspace}", f"PBS_JOBID={JOB}",
                   f"PBS_NODEFILE=/var/spool/pbs/aux/{JOB}", "PBS_MOMPORT=15003",
                   "PBS_ENVIRONMENT=PBS_INTERACTIVE", f"TRAIN_END_AT={timing['end_epoch']}"]
    command = [*environment, "/opt/pbs/bin/pbs_attach", "-j", JOB, "-s", "/usr/bin/bash", "-l", str(source / "run.sh")]
    return ("nohup " + shlex.join(command) + " >" + shlex.quote(str(root / "control" / "launch.log"))
            + " 2>&1 </dev/null & printf '%s\\n' \"$!\"")


def watch(workspace, source):
    root = workspace / "experiment191" / "artifacts" / f"job-{JOB}"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "watch.lock").mkdir()
    ready = json.loads((source / "READY.json").read_text())
    if not ready["tests_passed"] or any(digest(source / name) != value for name, value in ready["files"].items()):
        raise ValueError("Deployment changed after its tests passed")
    parent = workspace / "experiment19/artifacts/job-286615.hunter-pbs01/checkpoints/step_00000328_000000104960"
    verify_parent(parent)
    previous_state, launched, offset, heartbeat = None, False, 0, 0
    while True:
        result = subprocess.run(["/opt/pbs/bin/qstat", "-xf", "-F", "json", JOB],
                                check=True, capture_output=True, text=True, timeout=30)
        job = json.loads(result.stdout)["Jobs"][JOB]
        timing = allocation(job)
        now = time.time()
        if job["job_state"] != previous_state or now - heartbeat >= 60:
            report = {"event": "scheduler", "job": JOB, "state": job["job_state"],
                      "comment": job.get("comment"), "time": now, "launch_requested": launched}
            write_json(control / "status.json", report)
            print(json.dumps(report), flush=True)
            previous_state, heartbeat = job["job_state"], now
        if timing is not None and not launched:
            if now + 900 >= timing["stop_epoch"]:
                raise RuntimeError("Allocation has insufficient time remaining")
            (control / "launch.lock").mkdir()
            write_json(control / "deadlines.json", {"job": JOB, **timing})
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                     timing["hosts"][0], launch_command(workspace, source, root, timing)],
                                    env={**os.environ, "PBS_JOBID": JOB}, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"Authorized job-bound launch failed: {result.stderr.strip()}")
            launched = True
            report = {"event": "launch_requested", "job": JOB, "master": timing["hosts"][0],
                      "launcher_pid": result.stdout.strip(), "time": time.time(), **timing}
            write_json(control / "launch.json", report)
            print(json.dumps(report), flush=True)
        logfile = control / "launch.log"
        if logfile.exists():
            with logfile.open() as stream:
                stream.seek(offset)
                text = stream.read()
                offset = stream.tell()
            if text:
                print(text, end="", flush=True)
                if "Traceback (most recent call last)" in text:
                    raise RuntimeError("Training reported a traceback; inspect per-rank logs")
        if (root / "TRAINING_DONE.json").is_file():
            print(json.dumps({"event": "watch_complete", "job": JOB,
                              "training": json.loads((root / "TRAINING_DONE.json").read_text())}), flush=True)
            return
        if job["job_state"] == "F":
            raise RuntimeError("Allocation finished without a completed training report")
        threading.Event().wait(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        watch(arguments.workspace, arguments.source)
    except Exception as error:
        report = {"event": "watch_failed", "job": JOB, "error": str(error), "time": time.time()}
        write_json(arguments.workspace / "experiment191" / "artifacts" / f"job-{JOB}" / "control" / "failure.json", report)
        print(json.dumps(report), flush=True)
        raise