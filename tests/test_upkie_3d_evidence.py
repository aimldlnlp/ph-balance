"""Failure-aware metric gates and independent evidence audit checks."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
if HAS_MUJOCO:
    from experiments.upkie_3d_pilot import Case, measure
    from experiments.evaluate_upkie_3d import summary
    from experiments.analyze_results import audit
    from experiments.reproduce_upkie_3d_samples import compare_metrics


@unittest.skipUnless(HAS_MUJOCO, "Install MuJoCo for integrated evidence tests")
class EvidenceTests(unittest.TestCase):
    def test_reproduction_allows_only_integral_roundoff(self):
        original = dict(success=True, wheel_absolute_torque_integral=.8, all_absolute_torque_integral=130.)
        repeated = dict(original, wheel_absolute_torque_integral=np.nextafter(.8, 0))
        self.assertLess(abs(compare_metrics(original, repeated)["wheel_absolute_torque_integral"]), 1e-12)
        with self.assertRaises(ValueError):
            compare_metrics(original, dict(repeated, success=False))
        with self.assertRaises(ValueError):
            compare_metrics(original, dict(repeated, all_absolute_torque_integral=130.000001))

    def history(self):
        h = np.zeros((12001, 19))
        h[:, 0] = np.arange(len(h))*.005
        h[:, 6:8] = .44
        h[:, 12:14] = .1
        h[:, 18] = h[:, 0]-.01
        return h

    def test_success_needs_complete_horizon_and_all_position_gates(self):
        h = self.history()
        case = Case(pitch=0, heading=0)
        self.assertTrue(measure(h, case, None)["success"])
        self.assertFalse(measure(h[:-1], case, None)["success"])
        self.assertFalse(measure(h, case, "nonwheel_ground_contact:bracket")["success"])
        h[-1, 4] = .16
        self.assertIn("final_position", measure(h, case, None)["failed_metric_gates"])
        h[-1, 4], h[500, 4] = 0., .51
        self.assertIn("excursion", measure(h, case, None)["failed_metric_gates"])

    def test_persistent_settling_does_not_ignore_late_violation(self):
        h = self.history()
        h[-1, 2] = np.deg2rad(2.1)
        result = measure(h, Case(pitch=0, heading=0), None)
        self.assertFalse(result["success"])
        self.assertIsNone(result["settling_s"][-1])

    def test_failure_effort_is_not_pooled_with_full_horizon_successes(self):
        rows = []
        for hybrid in (True, False):
            for seed, good, effort in ((1, True, 10.), (2, False, .01)):
                rows.append(dict(episode=f"{'hybrid' if hybrid else 'lqr'}/{seed}", case={"seed": seed}, hybrid=hybrid,
                                 success=good, failure=None if good else "contact", wheel_absolute_torque_integral=effort))
        s = summary(rows)
        self.assertEqual(s["hybrid"]["successful_mean_wheel_effort"], 10.)
        self.assertEqual(s["hybrid"]["failures"], {"contact": 1})
        self.assertEqual(s["ablation"]["common_success_pairs"], 1)
        self.assertFalse(s["ablation"]["point_hypothesis_met"])

    def fixture(self, directory):
        protocol = dict(robust_cases=[asdict(Case(seed=1, pitch=0, heading=0))], combined_cases=[asdict(Case(seed=2, pitch=0, heading=0))],
                        locked_validation=[asdict(Case(seed=3, pitch=0, heading=0))])
        (directory/"protocol.json").write_text(json.dumps(protocol))
        rows = []
        for group, seed in (("hybrid", 1), ("lqr", 1), ("combined", 2), ("locked_validation", 3)):
            h = self.history()
            row = measure(h, Case(seed=seed, pitch=0, heading=0), None)
            row.update(episode=f"{group}/{seed}", hybrid=group != "lqr")
            p = directory/row["episode"]
            p.mkdir(parents=True)
            qpos = np.zeros((len(h), 7))
            qpos[:, 3] = 1.
            observations = np.zeros((len(h), 20))
            observations[:, 0], observations[:, 1] = h[:, 0], h[:, 0]+.01
            np.savez_compressed(p/"trajectory.npz", history=h, qpos=qpos, qvel=np.zeros((len(h), 6)))
            np.savez_compressed(p/"observations.npz", observations=observations)
            rows.append(row)
        return rows

    def test_audit_checks_denominators_and_unavailable_observations(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = self.fixture(directory)
            self.assertEqual(audit(directory, rows, summary(rows))["episodes_checked"], 4)
            with self.assertRaisesRegex(ValueError, "Missing, duplicate"):
                audit(directory, rows[:-1], summary(rows[:-1]))
            rows[0]["case"]["friction"] = .1
            with self.assertRaisesRegex(ValueError, "conditions differ"):
                audit(directory, rows, summary(rows))
            rows[0]["case"]["friction"] = .8
            p = directory/"hybrid/1/trajectory.npz"
            native = np.load(p)
            h, qp, qv = native["history"].copy(), native["qpos"].copy(), native["qvel"].copy()
            native.close()
            h[5, 18] = h[5, 0]
            np.savez_compressed(p, history=h, qpos=qp, qvel=qv)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                audit(directory, rows, summary(rows))


if __name__ == "__main__":
    unittest.main()
