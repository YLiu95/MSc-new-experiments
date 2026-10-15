from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import math
import re
import subprocess


def slurm_time(output: str, key: str, tzinfo) -> datetime:
    found = re.search(rf"\b{key}=(\S+)", output)
    if not found or found.group(1) in ("Unknown", "N/A", "None"):
        raise ValueError(f"Slurm did not report a usable {key}")
    return datetime.fromisoformat(found.group(1)).replace(tzinfo=tzinfo)


def cutoff(reservation_end: datetime, job_start: datetime, job_end: datetime,
           now: datetime) -> tuple[datetime, datetime]:
    hard_end = min(reservation_end, job_end)
    wall_seconds = (job_end - job_start).total_seconds()
    reserve_seconds = max(2400, math.ceil(0.15 * wall_seconds))
    backup_by = hard_end - timedelta(minutes=20)
    stop_at = min(hard_end - timedelta(seconds=reserve_seconds), backup_by - timedelta(minutes=10))
    if stop_at <= now or backup_by <= now:
        raise ValueError("No time remains to train, stop, verify backups, and leave a 20-minute margin")
    return stop_at, backup_by


def live_deadlines(job_id: str, reservation: str = "training") -> tuple[datetime, datetime]:
    tzinfo = datetime.now().astimezone().tzinfo
    reservation_output = subprocess.check_output(["scontrol", "show", "reservation", reservation], text=True)
    job_output = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    if "State=ACTIVE" not in reservation_output:
        raise ValueError("Selected GPU reservation is no longer active")
    return cutoff(slurm_time(reservation_output, "EndTime", tzinfo),
                  slurm_time(job_output, "StartTime", tzinfo),
                  slurm_time(job_output, "EndTime", tzinfo), datetime.now().astimezone())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reservation", default="training")
    arguments = parser.parse_args()
    stop_at, backup_by = live_deadlines(arguments.job_id, arguments.reservation)
    print(stop_at.isoformat())
    print(backup_by.isoformat())


if __name__ == "__main__":
    main()