"""Unit tests for the nonlinear self-balancing robot model."""

import unittest
from dataclasses import replace

from src.models import (
    RobotParameters,
    RobotState,
    equations_of_motion,
    mass_matrix,
    state_derivative,
)
from src.models.self_balancing_robot import _solve_mass_matrix


def make_parameters() -> RobotParameters:
    """Return a small deterministic parameter set for unit tests."""

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


class SelfBalancingRobotTests(unittest.TestCase):
    """Check dimensions, equilibrium, and basic physical behavior."""

    def test_mass_matrix_is_symmetric_and_two_by_two(self) -> None:
        """The configuration mass matrix has the expected dimensions."""

        matrix = mass_matrix(0.15, make_parameters())

        self.assertEqual(len(matrix), 2)
        self.assertTrue(all(len(row) == 2 for row in matrix))
        self.assertAlmostEqual(matrix[0][1], matrix[1][0])

    def test_upright_rest_state_is_an_equilibrium(self) -> None:
        """Zero torque at upright rest produces zero acceleration."""

        state = RobotState(0.0, 0.0, 0.0, 0.0)

        derivative = state_derivative(state, 0.0, make_parameters())

        self.assertEqual(derivative, RobotState(0.0, 0.0, 0.0, 0.0))

    def test_upright_gravity_direction_is_unstable(self) -> None:
        """A positive lean accelerates farther from upright without torque."""

        state = RobotState(0.1, 0.0, 0.0, 0.0)

        _, theta_acceleration = equations_of_motion(
            state,
            0.0,
            make_parameters(),
        )

        self.assertGreater(theta_acceleration, 0.0)

    def test_external_body_torque_changes_pitch_acceleration(self) -> None:
        """A generalized body disturbance enters the pitch equation."""

        parameters = make_parameters()
        state = RobotState(0.0, 0.0, 0.0, 0.0)

        without_disturbance = equations_of_motion(state, 0.0, parameters)
        with_disturbance = equations_of_motion(
            state,
            0.0,
            parameters,
            external_torque=0.1,
        )

        self.assertEqual(without_disturbance[0], 0.0)
        self.assertNotEqual(with_disturbance[1], without_disturbance[1])

    def test_state_derivative_has_four_components(self) -> None:
        """The derivative preserves the required four-state representation."""

        derivative = state_derivative(
            RobotState(0.05, 0.1, 0.2, -0.1),
            0.02,
            make_parameters(),
        )

        self.assertEqual(len(derivative.as_tuple()), 4)

    def test_invalid_physical_parameter_is_rejected(self) -> None:
        """A zero wheel radius cannot define the actuator mapping."""

        parameters = replace(make_parameters(), wheel_radius=0.0)

        with self.assertRaises(ValueError):
            mass_matrix(0.0, parameters)

    def test_mass_matrix_dimension_validation(self) -> None:
        """The internal solver rejects malformed generalized dimensions."""

        with self.assertRaises(ValueError):
            _solve_mass_matrix(((1.0,), (0.0, 1.0)), (0.0, 0.0))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
