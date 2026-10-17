import time

from ranking17.offline import quota_available


PARAMETERS = 1069302417
RESERVE = 8 * 1024**3


def storage_status(root):
    checkpoint = PARAMETERS * 16
    return {"free_bytes": quota_available(root), "startup_required_bytes": 6 * checkpoint + RESERVE,
            "operating_required_bytes": 4 * checkpoint + RESERVE,
            "save_required_bytes": 2 * checkpoint + RESERVE, "time": time.time()}


def require_headroom(root, phase="startup"):
    if phase not in ("startup", "save"):
        raise ValueError("Unknown storage admission phase")
    status = storage_status(root)
    if status["free_bytes"] < status[f"{phase}_required_bytes"]:
        raise RuntimeError(f"Insufficient shared quota for both arms ({phase}): {status}")
    return status