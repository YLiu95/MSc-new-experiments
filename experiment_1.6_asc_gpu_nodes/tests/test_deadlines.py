from datetime import datetime, timedelta, timezone

import pytest

from ranker.deadlines import cutoff, slurm_time
from ranker.launch import request


def test_backup_verified_twenty_minutes_before_earlier_shutdown():
    zone = timezone(timedelta(hours=2))
    started = datetime(2026, 9, 29, 18, tzinfo=zone)
    reservation = datetime(2026, 9, 30, 8, tzinfo=zone)
    job_end = datetime(2026, 9, 30, 7, 30, tzinfo=zone)
    stop, backup_by = cutoff(reservation, started, job_end, started + timedelta(hours=1))
    assert backup_by == job_end - timedelta(minutes=20)
    assert stop <= job_end - timedelta(hours=2)
    with pytest.raises(ValueError):
        cutoff(reservation, started, job_end, job_end - timedelta(minutes=20))
    assert slurm_time("StartTime=2026-09-29T18:00:00 EndTime=2026-09-30T08:00:00", "EndTime", zone) == reservation


def test_batch_request_has_time_limit_deadline_and_no_wait_flag():
    zone = timezone(timedelta(hours=2))
    now = datetime(2026, 9, 29, 18, tzinfo=zone)
    end = datetime(2026, 9, 30, 8, tzinfo=zone)
    command = request(3, 600, 2, now=now, end=end)
    assert "--nodes=3" in command and "--gpus-per-node=8" in command
    assert "--time=120" in command and "--deadline=2026-09-29T20:10:00" in command
    assert "--export=NONE" in command and "--wait" not in command
    with pytest.raises(ValueError):
        request(3, 600, 6, now=now, end=end)