import unittest

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
        for nodes in (1, 2, 3, 4):
            self.assertEqual(validate_nodes(nodes), nodes)
        for nodes in (0, 5, True):
            with self.assertRaises(ValueError):
                validate_nodes(nodes)

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


if __name__ == "__main__":
    unittest.main()