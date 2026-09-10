"""Unit tests for the upright-relative Hamiltonian."""

import unittest
from math import isclose

from src.models import (
    RobotParameters,
    RobotState,
    energy_gradient,
    potential_energy,
    physical_energy_gradient,
    physical_potential_energy,
    physical_total_energy,
    total_energy,
)
from src.models.self_balancing_robot import mass_matrix


def make_parameters() -> RobotParameters:
    """Return deterministic parameters for energy tests."""

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


def state_from_canonical(
    canonical_state: tuple[float, float, float, float],
    parameters: RobotParameters,
) -> RobotState:
    """Convert ``[s, theta, p_s, p_theta]`` to the project state."""

    wheel_position, theta, p_s, p_theta = canonical_state
    matrix = mass_matrix(theta, parameters)
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    wheel_velocity = (d * p_s - b * p_theta) / determinant
    theta_dot = (-c * p_s + a * p_theta) / determinant
    return RobotState(theta, theta_dot, wheel_position, wheel_velocity)


class HamiltonianTests(unittest.TestCase):
    """Check energy shape and canonical gradient behavior."""

    def test_equilibrium_is_zero_energy(self) -> None:
        """Upright rest is the minimum energy reference."""

        state = RobotState(0.0, 0.0, 0.0, 0.0)

        self.assertAlmostEqual(total_energy(state, make_parameters()), 0.0)

    def test_energy_increases_with_angle_displacement(self) -> None:
        """Static energy grows as the angle moves away from upright."""

        parameters = make_parameters()
        zero = RobotState(0.0, 0.0, 0.0, 0.0)
        displaced = RobotState(0.2, 0.0, 0.0, 0.0)

        self.assertGreater(
            potential_energy(displaced, parameters),
            potential_energy(zero, parameters),
        )
        self.assertGreaterEqual(total_energy(displaced, parameters), 0.0)

    def test_physical_potential_has_upright_maximum(self) -> None:
        """The plant gravity is unstable at the upright equilibrium."""

        parameters = make_parameters()
        zero = RobotState(0.0, 0.0, 0.0, 0.0)
        displaced = RobotState(0.2, 0.0, 0.0, 0.0)

        self.assertGreater(
            physical_potential_energy(zero, parameters),
            physical_potential_energy(displaced, parameters),
        )

    def test_physical_gradient_matches_central_difference(self) -> None:
        """The physical Hamiltonian gradient has the plant gravity sign."""

        parameters = make_parameters()
        canonical_state = (0.1, 0.2, 0.03, -0.04)
        state = state_from_canonical(canonical_state, parameters)
        analytic = physical_energy_gradient(state, parameters)
        step = 1.0e-6
        numeric: list[float] = []

        for index in range(4):
            plus = list(canonical_state)
            minus = list(canonical_state)
            plus[index] += step
            minus[index] -= step
            numeric.append(
                (
                    physical_total_energy(
                        state_from_canonical(tuple(plus), parameters),
                        parameters,
                    )
                    - physical_total_energy(
                        state_from_canonical(tuple(minus), parameters),
                        parameters,
                    )
                )
                / (2.0 * step)
            )

        for expected, actual in zip(analytic, numeric):
            self.assertTrue(isclose(expected, actual, rel_tol=1.0e-5, abs_tol=1.0e-7))

    def test_gradient_matches_central_difference(self) -> None:
        """The analytic canonical gradient agrees with finite differences."""

        parameters = make_parameters()
        canonical_state = (0.1, 0.2, 0.03, -0.04)
        state = state_from_canonical(canonical_state, parameters)
        analytic = energy_gradient(state, parameters)
        step = 1.0e-6
        numeric: list[float] = []

        for index in range(4):
            plus = list(canonical_state)
            minus = list(canonical_state)
            plus[index] += step
            minus[index] -= step
            numeric.append(
                (
                    total_energy(state_from_canonical(tuple(plus), parameters), parameters)
                    - total_energy(state_from_canonical(tuple(minus), parameters), parameters)
                )
                / (2.0 * step)
            )

        for expected, actual in zip(analytic, numeric):
            self.assertTrue(isclose(expected, actual, rel_tol=1.0e-5, abs_tol=1.0e-7))


if __name__ == "__main__":
    unittest.main()
