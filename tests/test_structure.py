"""Smoke test for the initial PH-Balance package structure."""

import unittest

from src.controllers import PIDController, PortHamiltonianController
from src.models import RobotParameters, RobotState, state_derivative
from src.simulation import SimulationConfig, SimulationRunner


class StructureTests(unittest.TestCase):
    """Verify that the separated package boundaries are importable."""

    def test_skeleton_imports(self) -> None:
        """Instantiate the typed records exposed by the skeleton."""

        parameters = RobotParameters(
            body_mass=1.0,
            wheel_mass_per_wheel=0.1,
            body_inertia=0.1,
            wheel_inertia_per_wheel=0.01,
            center_of_mass_length=0.2,
            wheel_radius=0.05,
            gravity=9.81,
            translational_friction=0.0,
            rotational_friction=0.0,
        )
        state = RobotState(0.0, 0.0, 0.0, 0.0)
        runner = SimulationRunner(
            state_derivative,
            PIDController(angle_kp=0.0, angle_ki=0.0, angle_kd=0.0),
            parameters,
            SimulationConfig(1.0, 0.01),
        )

        self.assertEqual(state.as_tuple(), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(runner.config.timestep, 0.01)
        self.assertIsInstance(
            PIDController(angle_kp=0.0, angle_ki=0.0, angle_kd=0.0),
            PIDController,
        )
        self.assertIsInstance(
            PortHamiltonianController(energy_gain=0.0, damping_gain=0.0),
            PortHamiltonianController,
        )


if __name__ == "__main__":
    unittest.main()
