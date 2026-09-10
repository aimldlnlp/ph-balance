"""Unit tests for the solve_ivp simulation framework."""

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.models import RobotParameters, RobotState, state_derivative
from src.simulation import SimulationConfig, Simulator


def make_parameters() -> RobotParameters:
    """Return deterministic parameters for simulation tests."""

    return RobotParameters(
        body_mass=1.0,
        wheel_mass_per_wheel=0.1,
        body_inertia=0.1,
        wheel_inertia_per_wheel=0.01,
        center_of_mass_length=0.2,
        wheel_radius=0.05,
        gravity=9.81,
        translational_friction=0.02,
        rotational_friction=0.01,
    )


class ZeroController:
    """Memoryless test controller that applies no motor torque."""

    def compute_torque(self, time, state, parameters) -> float:
        """Return zero torque for every solver interval."""

        return 0.0


def legacy_model(state, motor_torque, parameters):
    """Adapt the original three-argument model contract for compatibility."""

    return state_derivative(state, motor_torque, parameters)


class CountingController(ZeroController):
    """Record controller sampling to verify zero-order hold behavior."""

    def __init__(self) -> None:
        self.calls = 0

    def compute_torque(self, time, state, parameters) -> float:
        """Count one call per output interval."""

        self.calls += 1
        return super().compute_torque(time, state, parameters)


class SimulatorTests(unittest.TestCase):
    """Check integration, history storage, and CSV export."""

    def make_simulator(self, controller) -> Simulator:
        """Build a short deterministic simulator."""

        return Simulator(
            model=state_derivative,
            controller=controller,
            parameters=make_parameters(),
            config=SimulationConfig(
                duration=0.1,
                timestep=0.02,
                relative_tolerance=1.0e-9,
                absolute_tolerance=1.0e-11,
            ),
        )

    def test_run_stores_all_required_histories(self) -> None:
        """The result contains aligned time, state, input, and energy data."""

        result = self.make_simulator(ZeroController()).run(
            RobotState(0.05, 0.0, 0.0, 0.0)
        )

        self.assertEqual(result.time.shape, (6,))
        self.assertEqual(result.state_history.shape, (6, 4))
        self.assertEqual(result.control_input.shape, (6,))
        self.assertEqual(result.energy.shape, (6,))
        self.assertTrue(np.isfinite(result.state_history).all())
        self.assertTrue(np.isfinite(result.energy).all())

    def test_physical_energy_is_conserved_without_input_or_friction(self) -> None:
        """The recorded plant Hamiltonian stays constant in the ideal case."""

        parameters = replace(
            make_parameters(),
            translational_friction=0.0,
            rotational_friction=0.0,
        )
        result = Simulator(
            model=state_derivative,
            controller=ZeroController(),
            parameters=parameters,
            config=SimulationConfig(
                duration=0.4,
                timestep=0.002,
                relative_tolerance=1.0e-9,
                absolute_tolerance=1.0e-11,
            ),
        ).run(RobotState(0.2, 0.0, 0.0, 0.0))

        self.assertLess(float(np.ptp(result.energy)), 1.0e-7)

    def test_disturbance_is_held_and_recorded(self) -> None:
        """A sampled disturbance is applied per interval and exported."""

        result = Simulator(
            model=state_derivative,
            controller=ZeroController(),
            parameters=make_parameters(),
            config=SimulationConfig(duration=0.1, timestep=0.02),
            disturbance=lambda time: 0.25 if time >= 0.04 else 0.0,
        ).run(RobotState(0.0, 0.0, 0.0, 0.0))

        self.assertEqual(result.disturbance[0], 0.0)
        self.assertEqual(result.disturbance[2], 0.25)
        self.assertEqual(result.disturbance[-1], 0.25)

    def test_angle_limit_terminates_invalid_local_model_trajectory(self) -> None:
        """A configured fallover bound stops integration before full rotation."""

        result = Simulator(
            model=state_derivative,
            controller=ZeroController(),
            parameters=make_parameters(),
            config=SimulationConfig(
                duration=4.0,
                timestep=0.02,
                max_abs_angle=0.5,
            ),
        ).run(RobotState(0.2, 0.0, 0.0, 0.0))

        self.assertEqual(result.termination_reason, "angle_limit")
        self.assertLess(result.time[-1], 4.0)
        self.assertLessEqual(abs(result.state_history[-1, 0]), 0.5 + 1.0e-9)

    def test_legacy_three_argument_model_still_runs(self) -> None:
        """Existing zero-disturbance model callables remain supported."""

        result = Simulator(
            model=legacy_model,
            controller=ZeroController(),
            parameters=make_parameters(),
            config=SimulationConfig(duration=0.1, timestep=0.02),
        ).run(RobotState(0.0, 0.0, 0.0, 0.0))

        self.assertEqual(len(result.time), 6)

    def test_controller_is_sampled_once_per_interval(self) -> None:
        """Adaptive solver stages do not mutate controller sampling state."""

        controller = CountingController()

        self.make_simulator(controller).run(RobotState(0.0, 0.0, 0.0, 0.0))

        self.assertEqual(controller.calls, 5)

    def test_result_exports_csv(self) -> None:
        """CSV export contains the requested output columns and rows."""

        result = self.make_simulator(ZeroController()).run(
            RobotState(0.0, 0.0, 0.0, 0.0)
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            result.export_csv(path)
            with path.open(newline="", encoding="utf-8") as input_file:
                rows = list(csv.reader(input_file))

        self.assertEqual(
            rows[0],
            [
                "time",
                "theta",
                "theta_dot",
                "wheel_position",
                "wheel_velocity",
                "control_input",
                "disturbance_torque",
                "energy",
            ],
        )
        self.assertEqual(len(rows), 7)

    def test_invalid_config_is_rejected(self) -> None:
        """Non-positive integration settings fail before integration."""

        with self.assertRaises(ValueError):
            SimulationConfig(duration=1.0, timestep=0.0)


if __name__ == "__main__":
    unittest.main()
