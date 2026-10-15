from datetime import datetime, timezone


PRODUCTION_END = datetime.fromisoformat("2026-10-15T08:00:00+02:00").timestamp()
QUEUE_WAIT_SECONDS = 600
SHUTDOWN_RESERVE_SECONDS = 1200


def validate_nodes(nodes):
    if isinstance(nodes, bool) or nodes not in (1, 2, 3, 4, 6):
        raise ValueError("Hunter supports 1-4 or 6 MI300A nodes")
    return nodes


def production_nodes(free_nodes, requested=None, waive_fraction=False):
    if requested is not None:
        if isinstance(requested, bool) or requested not in (2, 4, 6):
            raise ValueError("Production requires 2, 4 or 6 MI300A nodes")
        if requested > free_nodes or (not waive_fraction and requested * 5 >= free_nodes):
            raise ValueError("Production request violates the free-node policy")
        return requested
    if waive_fraction:
        raise ValueError("A policy waiver requires an explicit node count")
    for nodes in (4, 2):
        if nodes * 5 < free_nodes:
            return nodes
    raise ValueError("Insufficient free MI300A nodes for a request below 20 percent")


def free_apu_nodes(payload):
    return sum(
        node.get("state") == "free"
        and not node.get("jobs")
        and "mi300a" in node.get("resources_available", {}).get("node_type", "").split(",")
        for node in payload["nodes"].values()
    )


def duration_seconds(value):
    hours, minutes, seconds = map(int, value.split(":"))
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("Invalid PBS walltime")
    return hours * 3600 + minutes * 60 + seconds


def walltime(seconds):
    if seconds <= 0:
        raise ValueError("No allocation time remains")
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def running_deadlines(job, now, reserve=SHUTDOWN_RESERVE_SECONDS):
    if job.get("job_state") != "R":
        raise ValueError("Deadline requires a running PBS allocation")
    started = job["stime"]
    if not isinstance(started, (int, float)):
        started = datetime.strptime(started, "%a %b %d %H:%M:%S %Y").timestamp()
    allocation_end = started + duration_seconds(job["Resource_List"]["walltime"])
    end = min(allocation_end, PRODUCTION_END)
    stop = end - reserve
    if stop <= now:
        raise ValueError("Insufficient allocation time for protected shutdown")
    return {"stop_epoch": stop, "backup_by_epoch": end,
            "allocation_start_epoch": started, "allocation_end_epoch": allocation_end,
            "absolute_deadline": datetime.fromtimestamp(PRODUCTION_END, timezone.utc).isoformat(),
            "shutdown_reserve_seconds": reserve}