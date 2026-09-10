"""Unit tests for experiment metric calculations."""

import unittest

import numpy as np

from experiments.metrics import compute_metrics
from src.simulation import SimulationResult


class ExperimentMetricsTests(unittest.TestCase):
    """Check tolerance-band and disturbance-recovery calculations."""

    def make_result(self) -> SimulationResult:
        """Return a small deterministic trajectory."""

        time = np.arange(4.0)
        states = np.zeros((4, 4))
        states[:, 0] = (0.2, 0.05, 0.01, 0.0)
        return SimulationResult(
            time=time,
            state_history=states,
            control_input=np.ones(4),
            energy=np.ones(4),
        )

    def test_settling_time_uses_persistent_tolerance_band(self) -> None:
        """Settling is the first sample that stays within two degrees."""

        metrics = compute_metrics(
            self.make_result(),
            experiment="nominal",
            controller="PID",
        )

        self.assertEqual(metrics.settling_time_s, 2.0)
        self.assertEqual(metrics.success, 1)
        self.assertEqual(metrics.stable, 1)

    def test_disturbance_recovery_time_is_measured_from_event(self) -> None:
        """Recovery begins at the configured disturbance timestamp."""

        metrics = compute_metrics(
            self.make_result(),
            experiment="disturbance",
            controller="PID",
            disturbance_time=1.0,
        )

        self.assertEqual(metrics.recovery_time_s, 1.0)

    def test_wheel_drift_is_final_absolute_position(self) -> None:
        """Wheel drift is measured against the zero-position target."""

        result = self.make_result()
        states = result.state_history.copy()
        states[-1, 2] = -0.25
        result = SimulationResult(
            time=result.time,
            state_history=states,
            control_input=result.control_input,
            energy=result.energy,
        )

        metrics = compute_metrics(result, "nominal", "Hybrid PH")

        self.assertEqual(metrics.wheel_drift_m, 0.25)

    def test_torque_effort_uses_zero_order_hold(self) -> None:
        """Effort weights held inputs by interval duration only."""

        result = SimulationResult(
            time=np.array([0.0, 0.5, 1.0]),
            state_history=np.zeros((3, 4)),
            control_input=np.array([2.0, 0.0, 0.0]),
            energy=np.array([1.0, 1.5, 1.0]),
        )

        metrics = compute_metrics(result, "nominal", "PID")

        self.assertEqual(metrics.integral_abs_torque_nm_s, 1.0)
        self.assertEqual(metrics.mean_abs_torque_nm, 1.0)
        self.assertAlmostEqual(metrics.rms_torque_nm, np.sqrt(2.0))
        self.assertAlmostEqual(metrics.energy_deviation_rms_j, 0.5 / np.sqrt(2.0))
        self.assertEqual(metrics.final_energy_deviation_j, 0.0)

    def test_early_termination_before_disturbance_is_recorded(self) -> None:
        """Early fallover remains measurable even before a disturbance event."""

        states = np.zeros((3, 4))
        states[:, 0] = (0.0, 0.1, 0.5)
        result = SimulationResult(
            time=np.array([0.0, 0.2, 0.4]),
            state_history=states,
            control_input=np.ones(3),
            energy=np.ones(3),
            termination_reason="angle_limit",
        )

        metrics = compute_metrics(
            result,
            "disturbance",
            "PH-V3",
            disturbance_time=5.0,
        )

        self.assertEqual(metrics.success, 0)
        self.assertEqual(metrics.termination_reason, "angle_limit")
        self.assertEqual(metrics.termination_time_s, 0.4)
        self.assertEqual(metrics.fallover_penalty, 1.0)
        self.assertTrue(np.isnan(metrics.recovery_time_s))
        self.assertEqual(metrics.integral_abs_torque_nm_s, 0.4)


if __name__ == "__main__":
    unittest.main()
