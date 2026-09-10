"""Unit tests for the Port-Hamiltonian representation and controller."""

import unittest
from math import sin

from src.controllers import (
    PortHamiltonianController,
    PortHamiltonianV2Controller,
    PortHamiltonianV3Controller,
    dissipation_matrix,
    input_matrix,
    port_hamiltonian_derivative,
    structure_matrix,
)
from src.models import (
    RobotParameters,
    RobotState,
    equations_of_motion,
    mass_matrix,
    physical_energy_gradient,
    total_energy,
)


def make_parameters() -> RobotParameters:
    """Return deterministic parameters for controller tests."""

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


class PortHamiltonianTests(unittest.TestCase):
    """Check matrix structure and the un-tuned control law."""

    def test_structure_matrix_is_skew_symmetric(self) -> None:
        """J has four dimensions and exact skew symmetry."""

        matrix = structure_matrix()

        self.assertEqual(len(matrix), 4)
        self.assertTrue(all(len(row) == 4 for row in matrix))
        for row in range(4):
            for column in range(4):
                self.assertEqual(matrix[row][column], -matrix[column][row])

    def test_dissipation_and_input_dimensions(self) -> None:
        """R and G match the four-dimensional canonical state."""

        parameters = make_parameters()
        dissipation = dissipation_matrix(parameters)
        actuation = input_matrix(parameters)

        self.assertEqual(len(dissipation), 4)
        self.assertTrue(all(len(row) == 4 for row in dissipation))
        self.assertEqual(len(actuation), 4)

    def test_energy_shaping_term(self) -> None:
        """The energy term follows -Ke*(H-Hd)."""

        parameters = make_parameters()
        state = RobotState(0.2, 0.0, 0.0, 0.0)
        controller = PortHamiltonianController(
            energy_gain=2.0,
            damping_gain=0.0,
            desired_energy=0.0,
        )

        expected = -2.0 * total_energy(state, parameters)

        self.assertAlmostEqual(
            controller.compute_torque(0.0, state, parameters),
            expected,
        )

    def test_damping_uses_actuator_port_velocity(self) -> None:
        """The vector velocity is projected to y=B^T*q_dot."""

        parameters = make_parameters()
        state = RobotState(0.0, 0.1, 0.0, 0.03)
        controller = PortHamiltonianController(
            energy_gain=0.0,
            damping_gain=4.0,
        )
        port_velocity = 0.03 / parameters.wheel_radius - 0.1

        self.assertAlmostEqual(
            controller.compute_torque(0.0, state, parameters),
            -4.0 * port_velocity,
        )

    def test_optional_torque_limit(self) -> None:
        """Configured saturation bounds the commanded torque."""

        controller = PortHamiltonianController(
            energy_gain=100.0,
            damping_gain=0.0,
            torque_limit=0.5,
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.3, 0.0, 0.0, 0.0),
            make_parameters(),
        )

        self.assertLessEqual(abs(torque), 0.5)

    def test_port_hamiltonian_equilibrium(self) -> None:
        """Zero input at upright rest produces zero canonical derivative."""

        derivative = port_hamiltonian_derivative(
            RobotState(0.0, 0.0, 0.0, 0.0),
            0.0,
            make_parameters(),
        )

        self.assertEqual(derivative, (0.0, 0.0, 0.0, 0.0))

    def test_port_hamiltonian_matches_physical_dynamics(self) -> None:
        """Canonical pH dynamics agree with the nonlinear plant equations."""

        parameters = make_parameters()
        state = RobotState(0.2, 0.3, 0.1, -0.2)
        motor_torque = 0.04
        external_torque = 0.02
        wheel_acceleration, theta_acceleration = equations_of_motion(
            state,
            motor_torque,
            parameters,
            external_torque,
        )
        matrix = mass_matrix(state.theta, parameters)
        coupling = parameters.body_mass * parameters.center_of_mass_length
        mass_derivative_off_diagonal = (
            -coupling * sin(state.theta) * state.theta_dot
        )
        expected_p_s_dot = (
            matrix[0][0] * wheel_acceleration
            + matrix[0][1] * theta_acceleration
            + mass_derivative_off_diagonal * state.theta_dot
        )
        expected_p_theta_dot = (
            matrix[1][0] * wheel_acceleration
            + matrix[1][1] * theta_acceleration
            + mass_derivative_off_diagonal * state.wheel_velocity
        )

        derivative = port_hamiltonian_derivative(
            state,
            motor_torque,
            parameters,
            external_torque,
        )

        self.assertEqual(derivative[:2], (state.wheel_velocity, state.theta_dot))
        self.assertAlmostEqual(derivative[2], expected_p_s_dot)
        self.assertAlmostEqual(derivative[3], expected_p_theta_dot)

    def test_physical_energy_balance_has_expected_power_terms(self) -> None:
        """The pH model dissipates energy and receives motor power only."""

        parameters = make_parameters()
        state = RobotState(0.2, 0.3, 0.1, -0.2)
        motor_torque = 0.04
        gradient = physical_energy_gradient(state, parameters)
        derivative = port_hamiltonian_derivative(
            state,
            motor_torque,
            parameters,
        )
        energy_rate = sum(
            gradient[index] * derivative[index] for index in range(4)
        )
        port_velocity = (
            state.wheel_velocity / parameters.wheel_radius - state.theta_dot
        )
        expected_rate = (
            -parameters.translational_friction * state.wheel_velocity**2
            - parameters.rotational_friction * state.theta_dot**2
            + motor_torque * port_velocity
        )

        self.assertAlmostEqual(energy_rate, expected_rate)

    def test_invalid_gain_is_rejected(self) -> None:
        """Negative gains are invalid for this dissipative controller."""

        with self.assertRaises(ValueError):
            PortHamiltonianController(energy_gain=-1.0, damping_gain=0.0)

    def test_v2_output_direction(self) -> None:
        """PH-V2 commands opposite restoring torque for opposite lean."""

        parameters = make_parameters()
        controller = PortHamiltonianV2Controller(
            energy_gain=0.0,
            angle_gain=2.0,
            damping_gain=0.0,
        )

        positive_torque = controller.compute_torque(
            0.0,
            RobotState(0.2, 0.0, 0.0, 0.0),
            parameters,
        )
        negative_torque = controller.compute_torque(
            0.0,
            RobotState(-0.2, 0.0, 0.0, 0.0),
            parameters,
        )

        self.assertGreater(positive_torque, 0.0)
        self.assertLess(negative_torque, 0.0)
        self.assertAlmostEqual(positive_torque, -negative_torque)

    def test_v2_zero_equilibrium_behavior(self) -> None:
        """Upright rest produces zero PH-V2 torque when desired energy is zero."""

        controller = PortHamiltonianV2Controller(
            energy_gain=2.0,
            angle_gain=3.0,
            damping_gain=4.0,
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.0, 0.0, 0.0, 0.0),
            make_parameters(),
        )

        self.assertEqual(torque, 0.0)

    def test_v2_parameter_configuration(self) -> None:
        """All PH-V2 gains and the desired energy affect the output."""

        parameters = make_parameters()
        state = RobotState(0.2, 0.1, 0.0, 0.03)
        controller = PortHamiltonianV2Controller(
            energy_gain=2.0,
            angle_gain=3.0,
            damping_gain=4.0,
            desired_energy=0.5,
        )
        energy_error = total_energy(state, parameters) - 0.5
        port_velocity = 0.03 / parameters.wheel_radius - 0.1
        expected = (
            -2.0 * energy_error * port_velocity
            + 3.0 * sin(0.2)
            - 4.0 * port_velocity
        )

        self.assertAlmostEqual(
            controller.compute_torque(0.0, state, parameters),
            expected,
        )

    def test_v2_torque_saturation(self) -> None:
        """PH-V2 respects the configured motor torque limit."""

        controller = PortHamiltonianV2Controller(
            energy_gain=0.0,
            angle_gain=100.0,
            damping_gain=0.0,
            torque_limit=0.5,
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.3, 0.0, 0.0, 0.0),
            make_parameters(),
        )

        self.assertLessEqual(abs(torque), 0.5)

    def test_v3_angle_and_position_feedback_direction(self) -> None:
        """PH-V3 feedback terms reverse with lean and wheel displacement."""

        parameters = make_parameters()
        controller = PortHamiltonianV3Controller(
            energy_gain=0.0,
            angle_gain=2.0,
            position_gain=3.0,
            damping_gain=0.0,
        )

        positive = controller.compute_torque(
            0.0,
            RobotState(0.2, 0.0, 0.1, 0.0),
            parameters,
        )
        negative = controller.compute_torque(
            0.0,
            RobotState(-0.2, 0.0, -0.1, 0.0),
            parameters,
        )

        self.assertGreater(positive, 0.0)
        self.assertLess(negative, 0.0)
        self.assertAlmostEqual(positive, -negative)

    def test_v3_zero_equilibrium_behavior(self) -> None:
        """Upright rest at zero wheel position produces zero torque."""

        controller = PortHamiltonianV3Controller(
            energy_gain=2.0,
            angle_gain=3.0,
            position_gain=4.0,
            damping_gain=5.0,
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.0, 0.0, 0.0, 0.0),
            make_parameters(),
        )

        self.assertEqual(torque, 0.0)

    def test_v3_parameter_configuration(self) -> None:
        """All PH-V3 gains and the desired energy affect the output."""

        parameters = make_parameters()
        state = RobotState(0.2, 0.1, 0.03, 0.02)
        controller = PortHamiltonianV3Controller(
            energy_gain=1.5,
            angle_gain=2.5,
            position_gain=3.0,
            damping_gain=4.0,
            desired_energy=0.5,
        )
        energy_error = total_energy(state, parameters) - 0.5
        port_velocity = 0.02 / parameters.wheel_radius - 0.1
        expected = (
            -1.5 * energy_error * port_velocity
            + 2.5 * sin(0.2)
            + 3.0 * 0.03
            + 4.0 * 0.02
        )

        self.assertAlmostEqual(
            controller.compute_torque(0.0, state, parameters),
            expected,
        )

    def test_v3_torque_saturation(self) -> None:
        """PH-V3 respects the configured motor torque limit."""

        controller = PortHamiltonianV3Controller(
            energy_gain=0.0,
            angle_gain=100.0,
            position_gain=100.0,
            damping_gain=100.0,
            torque_limit=0.5,
        )

        torque = controller.compute_torque(
            0.0,
            RobotState(0.3, 0.1, 0.2, 0.1),
            make_parameters(),
        )

        self.assertLessEqual(abs(torque), 0.5)

    def test_invalid_v3_gain_is_rejected(self) -> None:
        """Negative PH-V3 gains are rejected at construction time."""

        with self.assertRaises(ValueError):
            PortHamiltonianV3Controller(
                energy_gain=0.0,
                angle_gain=0.0,
                position_gain=-1.0,
                damping_gain=0.0,
            )


if __name__ == "__main__":
    unittest.main()
