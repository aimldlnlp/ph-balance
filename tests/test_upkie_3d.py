"""Physical invariants and causal boundaries of the integrated Upkie path."""
import unittest
import importlib.util
from dataclasses import replace

import numpy as np

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
if HAS_MUJOCO:
    import mujoco
    from experiments.upkie_3d_pilot import Case, measure, reference, schedule, run
    from src.simulation.upkie_3d import build_floating, floating_lqr
    from src.simulation.upkie_observer import Sensors, Observer, Observation
    from src.simulation.upkie_active_lqr import active_design, ActiveObserver
    from experiments.evaluate_upkie_3d import cases, wilson, summary
    from experiments.check_upkie_3d_geometry import floor_clearances
    from experiments.audit_upkie_3d_native import coordinates


@unittest.skipUnless(HAS_MUJOCO, "Install MuJoCo for integrated Upkie tests")
class FloatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = build_floating()

    def test_floating_base_and_initial_clearance(self):
        m = self.robot.model
        self.assertEqual(m.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.neq, 0)
        self.assertEqual(m.nv, 8)
        for pitch in (-10, 0, 10):
            d = self.robot.initial_data(pitch)
            self.assertEqual(self.robot.nonwheel_ground_contacts(d), [])
            self.assertGreater(d.ncon, 0)

    def test_torso_only_uncertainty_and_source_limits(self):
        m, p = self.robot.model, build_floating(torso_scale=1.2).model
        difference = p.body_mass-m.body_mass
        self.assertAlmostEqual(sum(difference), .2)
        self.assertEqual(np.count_nonzero(difference), 1)
        np.testing.assert_allclose(m.actuator_ctrlrange, [[-1.7, 1.7]]*2)
        active = build_floating(active=True)
        self.assertEqual(active.model.nv, 12)
        np.testing.assert_allclose(active.model.actuator_ctrlrange[2:], [[-16, 16]]*4)
        for name in ("left_hip", "left_knee", "right_hip", "right_knee"):
            joint = active.model.joint(name)
            np.testing.assert_allclose(joint.range, [active.limits[name][k] for k in ("lower", "upper")])

    def test_nonwheel_ground_contact_is_failure(self):
        d = self.robot.initial_data()
        d.qpos[2] -= .10
        mujoco.mj_forward(self.robot.model, d)
        self.assertTrue(self.robot.nonwheel_ground_contacts(d))

    def test_clearance_checker_distinguishes_release_from_bracket_strike(self):
        d = self.robot.initial_data(-10)
        self.assertGreater(min(floor_clearances(self.robot, d).values()), 0.)
        d.qpos[2] -= .01
        mujoco.mj_forward(self.robot.model, d)
        self.assertLess(min(floor_clearances(self.robot, d).values()), 0.)

    def test_native_pose_coordinates_follow_actual_root_translation(self):
        d = self.robot.initial_data()
        moved = d.qpos.copy()
        moved[0] += .1
        moved[2] += .04
        xyz = coordinates(self.robot, np.array([d.qpos.copy(), moved]))
        np.testing.assert_allclose(xyz[1]-xyz[0], [.1, 0., .04], atol=1e-12)

    def test_causal_sensor_delay_reproducible_and_noise_rms(self):
        a, b = Sensors(self.robot, 7, .5), Sensors(self.robot, 7, .5)
        d = self.robot.initial_data()
        for s in (a, b):
            s.sample(d, 0.)
            self.assertEqual(s.receive(.009), [])
        x, y = a.receive(.01)[0], b.receive(.01)[0]
        np.testing.assert_array_equal(x.attitude, y.attitude)
        observer = Observer()
        with self.assertRaises(ValueError):
            observer.update(x, .009)
        observer.update(x, .01)
        with self.assertRaises(ValueError):
            observer.update(x, .02)
        errors = []
        for k in range(3000):
            a.sample(d, k*.005)
            errors.append(a.pending.pop().attitude-self.robot.orientation(d))
        self.assertAlmostEqual(np.rad2deg(np.sqrt(np.mean(np.array(errors)**2))), .5, delta=.02)

    def test_predictor_uses_only_past_available_data(self):
        _, ad, bd = floating_lqr(self.robot, matrices=True)
        observer = Observer(ad, bd, self.robot.trim)
        observation = Observation(0., .01, np.array([0., self.robot.trim, 0.]), np.zeros(3), np.zeros(2), np.zeros(2))
        observer.command(0., .1)
        observer.command(.005, -.1)
        observer.update(observation, .01)
        estimate = observer.estimate(.01)
        expected = ad@(bd*.1)-bd*.1
        self.assertAlmostEqual(estimate[0][1]-self.robot.trim, expected[0])
        with self.assertRaises(ValueError):
            observer.estimate(-.01)

    def test_lqr_invalid_weights_and_local_stability(self):
        gain, ad, bd = floating_lqr(self.robot, matrices=True)
        self.assertLess(max(abs(np.linalg.eigvals(ad-np.outer(bd, gain)))), 1.)
        with self.assertRaises(ValueError):
            floating_lqr(self.robot, r=0)

    def test_short_physical_recovery_both_controllers(self):
        table = schedule()
        for hybrid in (True, False):
            _, h, _, _, metrics = run(Case(duration=4, pitch=-10, heading=0), table, hybrid=hybrid)
            self.assertIsNone(metrics["failure"])
            self.assertEqual(h[-1, 0], 4)
            self.assertLess(abs(np.rad2deg(h[-1, 2])), 2)
            self.assertTrue(np.all(abs(h[:, 12:14]) <= 1.7))
            self.assertTrue(np.all(h[h[:, 18] >= 0, 18] <= h[h[:, 18] >= 0, 0]-.01+1e-9))

    def test_no_collision_masks_hide_link_collisions(self):
        m = self.robot.model
        collision = m.geom_contype != 0
        self.assertTrue(np.all(m.geom_conaffinity[collision] != 0))

    def test_active_model_and_observer_equilibrium(self):
        robot = build_floating(active=True)
        design = active_design(robot)
        self.assertLess(max(design["pole_magnitudes"]), 1.)
        self.assertGreater(min(np.linalg.eigvalsh(design["reduced_mass"])), 0.)
        observer = ActiveObserver(design)
        np.testing.assert_allclose(observer.predict(design["equilibrium"], 0), design["equilibrium"])
        sensor = Sensors(robot, 1, 0.)
        sensor.sample(robot.initial_data(), 0.)
        packet = sensor.receive(.01)[0]
        with self.assertRaises(ValueError):
            observer.update(packet, .009)
        observer.update(packet, .01)
        self.assertTrue(np.isfinite(observer.active_state(.01)).all())

    def test_height_schedule_three_complete_four_cm_cycles(self):
        table = schedule(True)
        high = reference(0, table, True)[1]
        for start in (6, 22, 38):
            self.assertAlmostEqual(high-reference(start+3, table, True)[1], .04)
            self.assertAlmostEqual(reference(start+11, table, True)[1], high)
        for time in np.linspace(0, 60, 121):
            row, _ = reference(time, table, True)
            self.assertEqual(row.shape, (446,))
            self.assertTrue(np.isfinite(row).all())

    def test_seed_partitions_and_binomial_interval(self):
        tuning = {c.seed for c in cases(range(1000, 1008))}
        robust = {c.seed for c in cases(range(100000, 100200))}
        combined = {c.seed for c in cases(range(200000, 200100), True)}
        self.assertFalse(tuning & robust or robust & combined or tuning & combined)
        self.assertEqual(len(robust), 200)
        self.assertEqual(len(combined), 100)
        lo, hi = wilson(190, 200)
        self.assertLess(lo, .95)
        self.assertGreater(hi, .95)
        self.assertAlmostEqual(wilson(0, 200)[0], 0.)
        with self.assertRaises(ValueError):
            wilson(1, 0)


if __name__ == "__main__":
    unittest.main()
