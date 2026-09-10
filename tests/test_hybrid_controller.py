"""Unit tests for the hybrid Port-Hamiltonian controller."""

import unittest

from src.controllers import HybridPortHamiltonianController
from src.models import RobotParameters, RobotState, total_energy


def make_parameters() -> RobotParameters:
    """Return the nominal physical parameters used by controller tests."""

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


class HybridControllerTests(unittest.TestCase):
    """Check the hybrid control law and its safety boundaries."""

    def test_equilibrium_produces_zero_torque(self) -> None:
        """Zero energy error and zero state feedback preserve equilibrium."""

        controller = HybridPortHamiltonianController(
            energy_gain=1.0,
            damping_gain=2.0,
            stabilizer_gains=(-2.0, -3.0, -4.0, -5.0),
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.0, 0.0, 0.0, 0.0),
            make_parameters(),
        )

        self.assertEqual(torque, 0.0)

    def test_state_feedback_uses_project_state_order(self) -> None:
        """The stabilizer applies ``-Kx`` in the documented state order."""

        parameters = make_parameters()
        state = RobotState(0.2, -0.1, 0.03, 0.02)
        controller = HybridPortHamiltonianController(
            energy_gain=0.0,
            damping_gain=0.0,
            stabilizer_gains=(-2.0, -3.0, -4.0, -5.0),
        )

        expected = 2.0 * state.theta + 3.0 * state.theta_dot
        expected += 4.0 * state.wheel_position + 5.0 * state.wheel_velocity

        self.assertAlmostEqual(
            controller.compute_torque(0.0, state, parameters),
            expected,
        )

    def test_energy_and_damping_terms_are_combined(self) -> None:
        """PH energy shaping and port damping contribute to one torque."""

        parameters = make_parameters()
        state = RobotState(0.2, 0.1, 0.0, 0.03)
        controller = HybridPortHamiltonianController(
            energy_gain=2.0,
            damping_gain=4.0,
            stabilizer_gains=(0.0, 0.0, 0.0, 0.0),
            desired_energy=0.5,
        )
        energy_error = total_energy(state, parameters) - 0.5
        port_velocity = state.wheel_velocity / parameters.wheel_radius
        port_velocity -= state.theta_dot
        expected = -2.0 * energy_error * port_velocity
        expected -= 4.0 * port_velocity

        self.assertAlmostEqual(
            controller.compute_torque(0.0, state, parameters),
            expected,
        )

    def test_stabilizer_dimension_is_validated(self) -> None:
        """A state-feedback row must match the four-state model."""

        with self.assertRaises(ValueError):
            HybridPortHamiltonianController(
                energy_gain=0.0,
                damping_gain=0.0,
                stabilizer_gains=(1.0, 2.0, 3.0),
            )

    def test_single_torque_limit_bounds_the_combined_output(self) -> None:
        """Saturation is applied after PH and stabilizer terms are summed."""

        controller = HybridPortHamiltonianController(
            energy_gain=0.0,
            damping_gain=0.0,
            stabilizer_gains=(-100.0, -100.0, -100.0, -100.0),
            torque_limit=0.5,
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.3, 0.1, 0.2, 0.1),
            make_parameters(),
        )

        self.assertLessEqual(abs(torque), 0.5)


if __name__ == "__main__":
    unittest.main()
