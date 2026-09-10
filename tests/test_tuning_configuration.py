"""Checks for the selected controller tuning configuration."""

import unittest
from math import radians

from experiments.run_experiments import (
    PH_REFERENCE_GAINS,
    PID_GAINS,
    controller_builders,
)
from src.controllers.pid import PIDController
from src.models import RobotParameters, RobotState, state_derivative
from src.simulation import SimulationConfig, Simulator


class TuningConfigurationTests(unittest.TestCase):
    """Keep experiment defaults aligned with the recorded tuning result."""

    def test_selected_gains_build_both_controllers(self) -> None:
        """The selected gain dictionaries satisfy controller validation."""

        controllers = controller_builders()

        self.assertEqual(PID_GAINS["angle_kp"], 30.0)
        self.assertEqual(PID_GAINS["angle_ki"], 0.5)
        self.assertEqual(PID_GAINS["angle_kd"], 3.0)
        self.assertEqual(PH_REFERENCE_GAINS["energy_gain"], 0.1)
        self.assertEqual(PH_REFERENCE_GAINS["damping_gain"], 1.0)
        self.assertIsNotNone(controllers["PID"]())
        self.assertIsNotNone(controllers["Port-Hamiltonian"]())

    def test_selected_pid_locally_stabilizes_both_lean_directions(self) -> None:
        """The tuned PID returns small positive and negative perturbations."""

        parameters = RobotParameters(
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
        for angle in (-radians(1.0), radians(1.0)):
            result = Simulator(
                model=state_derivative,
                controller=PIDController(**PID_GAINS),
                parameters=parameters,
                config=SimulationConfig(
                    duration=2.0,
                    timestep=0.01,
                    max_abs_angle=0.5,
                ),
            ).run(RobotState(angle, 0.0, 0.0, 0.0))

            self.assertIsNone(result.termination_reason)
            self.assertLess(abs(result.state_history[-1, 0]), radians(0.1))


if __name__ == "__main__":
    unittest.main()
