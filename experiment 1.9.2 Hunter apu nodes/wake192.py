import argparse
from pathlib import Path
import time
from control192 import write_json
from publish192 import JOB, notify


def watch(root, endpoint, phase="allocation"):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f"{phase}.lock"
    lock.mkdir()
    record = {"job": JOB, "phase": phase, "transport": "current_session_terminal_notification"}
    try:
        write_json(root / f"{phase}.json", {**record, "state": "waiting", "time": time.time()})
        status = notify(endpoint, phase)
        write_json(root / f"{phase}.json", {**record, "state": "notified", "status": status, "time": time.time()})
    except Exception as error:
        write_json(root / f"{phase}.json", {**record, "state": "failed", "error": str(error), "time": time.time()})
        raise
    finally:
        lock.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Notify the originating terminal-tool session; never invoke Copilot CLI.")
    parser.add_argument("action", choices=("watch",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18773")
    parser.add_argument("--phase", choices=("allocation", "training", "completion"), default="allocation")
    arguments = parser.parse_args()
    watch(arguments.root, arguments.endpoint, arguments.phase)