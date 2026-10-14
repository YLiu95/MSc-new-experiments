import importlib
import json
import unittest

from ranker.data import write_json
from ranking17.recovery import backup, digest

from ranking17.pbs import (
    PRODUCTION_END, free_apu_nodes, production_nodes, running_deadlines,
    validate_nodes, walltime,
)


class PBSPolicyTests(unittest.TestCase):
    def test_strict_free_node_fraction(self):
        self.assertEqual(production_nodes(21), 4)
        self.assertEqual(production_nodes(20), 2)
        self.assertEqual(production_nodes(11), 2)
        with self.assertRaises(ValueError):
            production_nodes(10)

    def test_supported_layouts(self):
        for nodes in (1, 2, 3, 4, 6):
            self.assertEqual(validate_nodes(nodes), nodes)
        for nodes in (0, 5, 7, True):
            with self.assertRaises(ValueError):
                validate_nodes(nodes)

    def test_six_node_waiver_is_explicit_and_capacity_checked(self):
        self.assertEqual(production_nodes(22, requested=6, waive_fraction=True), 6)
        self.assertEqual(production_nodes(31, requested=6), 6)
        for available in (22, 30):
            with self.assertRaises(ValueError):
                production_nodes(available, requested=6)
        with self.assertRaises(ValueError):
            production_nodes(5, requested=6, waive_fraction=True)
        with self.assertRaises(ValueError):
            production_nodes(22, waive_fraction=True)
        for requested in (True, 1, 3, 5, 7):
            with self.assertRaises(ValueError):
                production_nodes(22, requested=requested, waive_fraction=True)

    def test_only_unoccupied_free_apus_count(self):
        apu = {"state": "free", "resources_available": {"node_type": "mi300a512gb912c,mi300a"}}
        payload = {"nodes": {"one": apu, "two": dict(apu, jobs=["other"]),
                             "three": dict(apu, state="offline"),
                             "four": {"state": "free", "resources_available": {"node_type": "genoa"}}}}
        self.assertEqual(free_apu_nodes(payload), 1)

    def test_deadline_uses_running_start_and_absolute_cap(self):
        started = PRODUCTION_END - 7200
        job = {"job_state": "R", "stime": started, "Resource_List": {"walltime": "04:00:00"}}
        result = running_deadlines(job, started)
        self.assertEqual(result["backup_by_epoch"], PRODUCTION_END)
        self.assertEqual(result["stop_epoch"], PRODUCTION_END - 1200)
        job["Resource_List"]["walltime"] = "01:00:00"
        self.assertEqual(running_deadlines(job, started)["stop_epoch"], started + 2400)

    def test_queued_and_expired_jobs_rejected(self):
        with self.assertRaises(ValueError):
            running_deadlines({"job_state": "Q"}, 0)
        job = {"job_state": "R", "stime": PRODUCTION_END, "Resource_List": {"walltime": "01:00:00"}}
        with self.assertRaises(ValueError):
            running_deadlines(job, PRODUCTION_END)

    def test_walltime_format(self):
        self.assertEqual(walltime(1500), "00:25:00")
        with self.assertRaises(ValueError):
            walltime(0)


def test_migration_submission_uses_isolated_source_and_records_waiver(tmp_path, monkeypatch):
    monkeypatch.setenv("HUNTER_WORKSPACE", str(tmp_path))
    launch = importlib.import_module("ranking17.launch")
    monkeypatch.setattr(launch, "WORKSPACE", tmp_path)
    monkeypatch.setattr(launch, "BASE", tmp_path / "artifacts")
    monkeypatch.setattr(launch, "SOURCE", tmp_path / "source-six")
    monkeypatch.setattr(launch.time, "time", lambda: PRODUCTION_END - 7200)
    control = tmp_path / "artifacts" / "control"
    control.mkdir(parents=True)
    write_json(control / "two-node-preflight.json", {"passed": True, "world": 8, "source_sha256": "parent"})
    monkeypatch.setattr(launch, "freeze", lambda *args: {
        "source_sha256": "new", "migration": {"parent_source_sha256": "parent"}})
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        if command[0] == "pbsnodes":
            return json.dumps({"nodes": {str(index): {"state": "free", "resources_available": {"node_type": "mi300a"}}
                                        for index in range(6)}})
        if command[0] == "qstat":
            return json.dumps({"Jobs": {"285983.hunter-pbs01": {"job_state": "F"}}})
        assert command[0] == "qsub"
        return "123.hunter-pbs01\n"

    monkeypatch.setattr(launch.subprocess, "check_output", check_output)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: None)
    checkpoint = tmp_path / "artifacts" / "job-285983.hunter-pbs01" / "checkpoints" / "step_185"
    launch.submit(6, "production", checkpoint, True)
    submitted = json.loads((control / "submission-123.hunter-pbs01.json").read_text())
    assert submitted["nodes"] == 6 and submitted["free_node_fraction_waived"] is True
    assert submitted["seconds"] == 7140
    assert submitted["queue_deadline_epoch"] - submitted["submitted_epoch"] == 600
    command = commands[-1]
    assert "select=6:node_type=mi300a:mpiprocs=1" in command
    assert f"HUNTER_SOURCE={tmp_path / 'source-six'}" in command[command.index("-v") + 1]
    assert f"HUNTER_MIGRATE_FROM={checkpoint.resolve()}" in command[command.index("-v") + 1]


def test_handoff_verifies_before_releasing_only_redundant_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HUNTER_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("BACKUP_ROOT", str(tmp_path / "backups"))
    launch = importlib.import_module("ranking17.launch")
    monkeypatch.setattr(launch, "BASE", tmp_path / "artifacts")
    monkeypatch.setattr(launch.time, "time", lambda: PRODUCTION_END - 7200)
    control = launch.BASE / "control"
    control.mkdir(parents=True)
    write_json(control / "two-node-preflight.json", {"passed": True, "world": 8, "source_sha256": "parent"})
    root = launch.BASE / "job-285983.hunter-pbs01"
    (root / "control").mkdir(parents=True)
    (root / "control" / "STOP").touch()
    checkpoint = root / "checkpoints" / "step_192"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "state.json"):
        write_json(checkpoint / name, {"step": 192})
    (checkpoint / "recovery.pt").write_bytes(b"preserve original training state")
    write_json(checkpoint / "manifest.json", {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                                               for path in checkpoint.iterdir()})
    (checkpoint / "COMPLETE").write_text(digest(checkpoint / "manifest.json") + "\n")
    write_json(root / "latest.json", {"directory": checkpoint.relative_to(root).as_posix()})
    copied = tmp_path / "backups" / root.name / checkpoint.name
    backup(checkpoint, copied)
    monkeypatch.setattr(launch, "freeze", lambda *args: {"migration": {"parent_source_sha256": "parent"}})
    monkeypatch.setattr(launch, "quota_available", lambda path: 100 * 1024**3)

    def check_output(command, **kwargs):
        if command[0] == "qstat":
            return json.dumps({"Jobs": {"parent": {"job_state": "F", "Exit_status": 0}}})
        return json.dumps({"nodes": {str(index): {"state": "free", "resources_available": {"node_type": "mi300a"}}
                                    for index in range(6)}})

    called = []

    def submit(nodes, mode, migrate_from, waive):
        assert checkpoint.exists() and not copied.exists()
        called.append((nodes, mode, migrate_from, waive))

    monkeypatch.setattr(launch.subprocess, "check_output", check_output)
    monkeypatch.setattr(launch, "submit", submit)
    launch.handoff(root, 6, True)
    assert called == [(6, "production", checkpoint.resolve(), True)]
    assert (root / "reports" / "retired-redundant-backup" / "manifest.json").is_file()
    assert (control / "six-node-handoff.json").is_file()


if __name__ == "__main__":
    unittest.main()