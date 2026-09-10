"""Robot state definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotState:
    """Planar robot state in SI units.

    The angle is measured from the upright equilibrium. Angles are in
    radians, positions in metres, and their derivatives in SI units.
    """

    theta: float
    theta_dot: float
    wheel_position: float
    wheel_velocity: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the state in the documented project ordering."""

        return (
            self.theta,
            self.theta_dot,
            self.wheel_position,
            self.wheel_velocity,
        )
