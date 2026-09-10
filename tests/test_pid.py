"""Unit tests for the PID comparison baseline."""

import unittest

from src.controllers import PIDController
from src.models import RobotParameters, RobotState


def make_parameters() -> RobotParameters:
    """Return parameters accepted by the shared controller interface."""

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


class PIDControllerTests(unittest.TestCase):
    """Check PID output, state handling, and anti-windup behavior."""

    def test_proportional_output_uses_angle_error(self) -> None:
        """The first sample has no integral or derivative contribution."""

        controller = PIDController(angle_kp=10.0, angle_ki=0.0, angle_kd=0.0)
        state = RobotState(0.1, 0.0, 0.0, 0.0)

        torque = controller.compute_torque(0.0, state, make_parameters())

        self.assertAlmostEqual(torque, 1.0)

    def test_integral_accumulates_with_elapsed_time(self) -> None:
        """The integral term uses controller timestamps."""

        controller = PIDController(angle_kp=0.0, angle_ki=2.0, angle_kd=0.0)
        state = RobotState(0.1, 0.0, 0.0, 0.0)
        parameters = make_parameters()

        controller.compute_torque(0.0, state, parameters)
        torque = controller.compute_torque(0.1, state, parameters)

        self.assertAlmostEqual(torque, 0.02)

    def test_position_loop_generates_angle_reference(self) -> None:
        """Wheel-position correction is included before angle control."""

        controller = PIDController(
            angle_kp=10.0,
            angle_ki=0.0,
            angle_kd=0.0,
            position_kp=0.5,
        )
        state = RobotState(0.0, 0.0, 0.2, 0.0)

        torque = controller.compute_torque(0.0, state, make_parameters())

        self.assertAlmostEqual(torque, -1.0)

    def test_integral_is_not_wound_up_when_saturated(self) -> None:
        """Conditional integration holds the integral against saturation."""

        controller = PIDController(
            angle_kp=10.0,
            angle_ki=5.0,
            angle_kd=0.0,
            torque_limit=1.0,
        )
        state = RobotState(1.0, 0.0, 0.0, 0.0)
        parameters = make_parameters()

        first = controller.compute_torque(0.0, state, parameters)
        second = controller.compute_torque(0.1, state, parameters)

        self.assertEqual(first, 1.0)
        self.assertEqual(second, 1.0)
        self.assertEqual(controller._angle_integral, 0.0)

    def test_reset_clears_controller_memory(self) -> None:
        """Reset makes a controller reusable for a new experiment."""

        controller = PIDController(angle_kp=0.0, angle_ki=1.0, angle_kd=0.0)
        state = RobotState(0.1, 0.0, 0.0, 0.0)
        parameters = make_parameters()

        controller.compute_torque(0.0, state, parameters)
        controller.compute_torque(0.1, state, parameters)
        controller.reset()

        self.assertEqual(controller.compute_torque(0.0, state, parameters), 0.0)

    def test_invalid_gain_is_rejected(self) -> None:
        """Negative gains are rejected at the configuration boundary."""

        with self.assertRaises(ValueError):
            PIDController(angle_kp=-1.0, angle_ki=0.0, angle_kd=0.0)


if __name__ == "__main__":
    unittest.main()
