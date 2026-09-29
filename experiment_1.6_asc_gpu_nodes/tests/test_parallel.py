import json
import subprocess
import sys


def test_two_rank_uneven_informative_and_padding_matches_reference():
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                             "-m", "ranker.parallel_check", "--device", "cpu"],
                            text=True, capture_output=True, timeout=120, check=True)
    reports = [json.loads(line)["workers"] for line in result.stdout.splitlines() if line.startswith("{")][0]
    assert sorted(entry["informative_on_rank"] for entry in reports) == [0, 2]
    assert all(entry["maximum_gradient_absolute_error"] <= 1e-6 for entry in reports)