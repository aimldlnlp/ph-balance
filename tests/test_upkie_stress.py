"""Causal transport, missing-step prediction, interpolation and physical work."""
import unittest
import importlib.util
from dataclasses import replace
import numpy as np
HAS_MUJOCO = importlib.util.find_spec('mujoco') is not None
if HAS_MUJOCO:
    from experiments.upkie_stress import (StressCase, StressSensors, GapObserver,
        GapActiveObserver, ActuatorObserver, actuator_schedule, actuator_design,
        fast_reference, run, schedule, reference, measure, unpack_design, integral_schedule, execute)
    from src.simulation.upkie_observer import Observation
    from src.simulation.upkie_3d import build_floating
    from src.simulation.upkie_active_lqr import active_design


@unittest.skipUnless(HAS_MUJOCO, 'Install MuJoCo for stress experiments')
class StressTests(unittest.TestCase):
    def test_invalid_parameters(self):
        for kwargs in ({"dropout": 1.}, {"motor_tau": -.1}, {"jitter": float("nan")}):
            with self.assertRaises(ValueError):
                StressCase(**kwargs)

    def test_transport_is_causal_and_reproducible(self):
        robot = build_floating()
        data = robot.initial_data()
        streams = []
        for _ in range(2):
            sensors = StressSensors(robot, StressCase(seed=7, jitter=.02, dropout=.3))
            delivered = []
            for t in np.arange(0, 1., .005):
                sensors.sample(data, t)
                packets = sensors.receive(t)
                self.assertTrue(all(p.available_at <= t+1e-12 for p in packets))
                delivered.extend(p.sampled_at for p in packets)
            self.assertTrue(np.all(np.diff(delivered) > 0))
            self.assertGreater(sensors.dropped, 0)
            self.assertGreater(sensors.stale, 0)
            streams.append(delivered)
        self.assertEqual(*streams)

    def test_gap_prediction_uses_all_past_commands(self):
        ad = np.eye(4)*1.01
        bd = np.ones(4)
        observer = GapObserver(ad, bd)
        p = Observation(0., 0., np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2))
        observer.update(p, 0.)
        for k, command in enumerate((1., 2., 3.)):
            observer.command(k*.005, command)
        observer.update(replace(p, sampled_at=.015, available_at=.015), .015)
        expected = ad@(ad@bd+bd*2)+bd*3
        expected[[0, 1, 3]] *= 1-np.array([.04, .7, .02])
        np.testing.assert_allclose(observer.sagittal, expected)
        with self.assertRaises(ValueError):
            observer.update(replace(p, sampled_at=1., available_at=1.), .02)

    def test_active_gap_prediction_and_future_guard(self):
        design = active_design(build_floating(active=True))
        observer = GapActiveObserver(design)
        p = Observation(0., 0., np.zeros(3), np.zeros(3), np.zeros(6), np.zeros(6))
        observer.update(p, 0.)
        observer.update(replace(p, sampled_at=.02, available_at=.02), .02)
        self.assertTrue(np.isfinite(observer.state).all())
        with self.assertRaises(ValueError):
            observer.update(replace(p, sampled_at=1., available_at=0.), .02)

    def test_fast_schedule_matches_original(self):
        table = schedule(True, r=.6)
        for t in np.linspace(0, 60, 123):
            a, height = fast_reference(t, table, True)
            b, original_height = reference(t, table, True)
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-12)
            self.assertEqual(height, original_height)

    def test_short_recovery_work_and_motor_limits(self):
        robot, h, qpos, qvel, metrics = run(StressCase(duration=2., pitch=5., dropout=.2, motor_tau=.005),
                                          schedule(r=.6), revised=True)
        self.assertIsNone(metrics['failure'])
        self.assertGreater(metrics['mechanical_total_positive_work_j'], 0)
        self.assertTrue(np.all(np.array(metrics['mechanical_positive_work_j']) >= np.array(metrics['mechanical_signed_work_j'])-1e-12))
        self.assertTrue(np.all(abs(robot.applied_commands) <= 1.7))
        self.assertEqual(len(h), len(qpos))
        self.assertTrue(np.isfinite(qvel).all())

    def test_lateral_event_has_own_recovery_window(self):
        h = np.zeros((12001, 19))
        h[:, 0] = np.arange(len(h))*.005
        h[:, 6:8] = .44
        h[(h[:, 0] >= 25)&(h[:, 0] < 25.5), 2] = np.deg2rad(3.)
        result = measure(h, StressCase(pitch=0, heading=0), None)
        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['settling_s'][2], .3)

    def test_augmented_motors_are_causal_and_locally_stable(self):
        table = actuator_schedule(schedule(True, r=.6))
        d, a = unpack_design(table[0]), actuator_design(table[0])
        self.assertLess(max(abs(np.linalg.eigvals(a['ad']-a['bd']@a['gain']))), 1.)
        observer = ActuatorObserver(d,a)
        torque = np.arange(6.)
        observer.command(0., torque)
        np.testing.assert_allclose(observer.motor_states[.005], (1-np.exp(-.005/.01))*torque)
        expected = d['equilibrium']+(a['ad']@np.r_[np.zeros(14),-d['feedforward']]+a['bd']@(torque-d['feedforward']))[:14]
        np.testing.assert_allclose(observer.predict(d['equilibrium'],0.), expected)

    def test_signed_work_matches_constant_hold_joint_displacement(self):
        robot,h,qpos,_,m = run(StressCase(duration=.2,pitch=3),schedule(r=.6))
        indices = robot.model.jnt_qposadr[robot.wheel_ids]
        expected = np.sum(h[:-1,12:14]*robot.model.actuator_gear[:,0]*np.diff(qpos[:,indices],axis=0),axis=0)
        np.testing.assert_allclose(m['mechanical_signed_work_j'],expected,atol=1e-12)
        np.testing.assert_allclose(m['actual_joint_effort_nm_s'], np.diff(h[:,0])@abs(h[:-1,12:14]), atol=1e-12)

    def test_integral_assisted_design_and_warmup_are_causal(self):
        table = integral_schedule(schedule(True,r=.6))
        self.assertEqual(table.shape,(8,566))
        robot,h,_,_,m = run(StressCase(duration=.1,active=True,pitch=4,warmup=.2,motor_tau=.01),
                            table,revised=True,integral_assisted=True)
        self.assertIsNone(m['failure'])
        self.assertLess(h[0,18],0.)
        self.assertNotEqual(h[0,8],0.)
        self.assertTrue(np.all(h[:,18] <= h[:,0]+1e-12))
        self.assertEqual(robot.sensor_history[0].sampled_at,-.2)

    def test_jobs_reject_path_escape_before_creating_output(self):
        from dataclasses import asdict
        with self.assertRaises(ValueError):
            execute('tmp/invalid-stress-job', [('../escape',asdict(StressCase()),'hybrid')],1)


if __name__ == '__main__':
    unittest.main()
