from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import pwd
import re
import subprocess

from .deadlines import slurm_time
from .data import write_json


def reservation_end() -> datetime:
    output = subprocess.check_output(["scontrol", "show", "reservation", "training"], text=True)
    if "State=ACTIVE" not in output:
        raise ValueError("Overnight GPU reservation is not active")
    return slurm_time(output, "EndTime", datetime.now().astimezone().tzinfo)


def request(nodes: int, wait_seconds: int, hours: float, forecast: bool = False,
            now: datetime | None = None, end: datetime | None = None) -> list[str]:
    if not 1 <= nodes <= 3 or wait_seconds < 0 or not 0 < hours <= 6 / nodes:
        raise ValueError("Screen request is limited to 1-3 arms/nodes and 48 total GPU-hours")
    now = now or datetime.now().astimezone()
    end = end or reservation_end()
    minutes_available = math.floor((end - now).total_seconds() / 60) - math.ceil(wait_seconds / 60) - 5
    minutes = min(math.floor(hours * 60), minutes_available)
    if minutes < 60:
        raise ValueError("A one-hour minimum job plus queue wait does not fit before reservation expiry")
    deadline = now + timedelta(seconds=wait_seconds, minutes=minutes)
    directory = Path(__file__).resolve().parents[1]
    root = Path("/net/tscratch/people") / pwd.getpwuid(os.getuid()).pw_name / "experiments" / "experiment_1.6"
    return ["sbatch", "--parsable", "--export=NONE", "--account=tutorial", "--partition=tutorial",
            "--reservation=training", f"--nodes={nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1",
            "--cpus-per-task=32", "--gpus-per-node=8", "--mem=256G", f"--time={minutes}",
            f"--deadline={deadline:%Y-%m-%dT%H:%M:%S}", "--signal=B:USR1@2400",
            f"--chdir={directory}", f"--output={root / 'control' / 'slurm-%j.out'}",
            f"--error={root / 'control' / 'slurm-%j.err'}",
            *(["--test-only"] if forecast else []), str(directory / "batch_entry.sh")]


def clean_environment() -> dict:
    environment = os.environ.copy()
    for key in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE", "HF_TOKEN",
                "GITHUB_TOKEN", "GH_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(key, None)
    return environment


def forecast(hours: float) -> None:
    checked_at = datetime.now().astimezone()
    end = reservation_end()
    for nodes in (1, 2, 3):
        command = request(nodes, 3600, min(hours, 6 / nodes), True, checked_at, end)
        result = subprocess.run(command, text=True, capture_output=True, env=clean_environment(), check=False)
        print(json.dumps({"checked_at": checked_at.isoformat(), "nodes": nodes, "gpus": nodes * 8,
                          "wall_minutes": next(option for option in command if option.startswith("--time=")),
                          "accepted": result.returncode == 0,
                          "prediction": (result.stdout + result.stderr).strip()}), flush=True)


def submit(nodes: int, wait_seconds: int, hours: float) -> str:
    command = request(nodes, wait_seconds, hours)
    root = Path("/net/tscratch/people") / pwd.getpwuid(os.getuid()).pw_name / "experiments" / "experiment_1.6"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = subprocess.run(command, text=True, capture_output=True, env=clean_environment(), check=True)
    job_id = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError(f"Submitted job but did not recognize Slurm ID: {result.stdout.strip()}")
    entry = {"job_id": job_id, "submitted_at": datetime.now().astimezone().isoformat(),
             "nodes": nodes, "gpus": 8 * nodes, "max_wait_seconds": wait_seconds,
             "requested_hours": hours, "slurm_args": [value for value in command if value.startswith("--")]}
    write_json(control / f"job-{job_id}.json", entry)
    with (control / "submissions.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("forecast", "submit"))
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--max-wait-seconds", type=int)
    parser.add_argument("--hours", type=float, default=2)
    arguments = parser.parse_args()
    if arguments.action == "forecast":
        forecast(arguments.hours)
    elif arguments.nodes is None or arguments.max_wait_seconds is None:
        parser.error("Submission requires the user's node count and maximum wait time")
    else:
        submit(arguments.nodes, arguments.max_wait_seconds, arguments.hours)


if __name__ == "__main__":
    main()