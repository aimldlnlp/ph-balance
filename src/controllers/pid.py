r"""Conventional PID baseline for comparison experiments.

The baseline uses a simple nested structure. The outer wheel-position term
generates an angle reference and the inner angle loop generates motor torque:

.. math::

   theta_ref = theta_d + K_{sp}(s-s_d) + K_{sd}\dot{s},

   e_theta = theta - theta_ref,

   u = K_p e_theta + K_i \int e_theta\,dt + K_d\dot{e}_theta.

The measured-minus-reference error convention matches the motor sign used by
the plant model, ``B = [1/r, -1]^T``. The integral uses conditional
integration: when torque is saturated, it is updated only if the error would
move the command back toward the allowed range.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Optional

from src.models.parameters import RobotParameters
from src.models.state import RobotState


@dataclass
class PIDController:
    """Configurable nested PID baseline with integral anti-windup."""

    angle_kp: float
    angle_ki: float
    angle_kd: float
    position_kp: float = 0.0
    position_kd: float = 0.0
    desired_angle: float = 0.0
    desired_position: float = 0.0
    torque_limit: Optional[float] = None
    angle_reference_limit: Optional[float] = None
    _angle_integral: float = field(default=0.0, init=False, repr=False)
    _previous_time: Optional[float] = field(default=None, init=False, repr=False)
    _previous_angle_error: Optional[float] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate gains and safety limits without tuning them."""

        gains = (
            ("angle_kp", self.angle_kp),
            ("angle_ki", self.angle_ki),
            ("angle_kd", self.angle_kd),
            ("position_kp", self.position_kp),
            ("position_kd", self.position_kd),
        )
        for name, value in gains:
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        for name, value in (
            ("desired_angle", self.desired_angle),
            ("desired_position", self.desired_position),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        for name, value in (
            ("torque_limit", self.torque_limit),
            ("angle_reference_limit", self.angle_reference_limit),
        ):
            if value is not None and (
                not isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")

    def reset(self) -> None:
        """Clear integral and timing state before a new experiment."""

        self._angle_integral = 0.0
        self._previous_time = None
        self._previous_angle_error = None

    def _saturate(self, torque: float) -> float:
        """Apply the configured actuator limit, if present."""

        if self.torque_limit is None:
            return torque
        return max(-self.torque_limit, min(self.torque_limit, torque))

    def _integral_update_allowed(self, raw_torque: float, error: float) -> bool:
        """Return whether conditional integration should accept this update."""

        if self.torque_limit is None:
            return True
        if -self.torque_limit <= raw_torque <= self.torque_limit:
            return True
        if raw_torque > self.torque_limit:
            return error < 0.0
        return error > 0.0

    def compute_torque(
        self,
        time: float,
        state: RobotState,
        parameters: RobotParameters,
    ) -> float:
        """Return the PID motor torque using the shared controller interface.

        ``parameters`` is accepted for interchangeability with the
        Port-Hamiltonian controller. This baseline only needs measured state,
        configured gains, and the controller clock.
        """

        if not isfinite(time):
            raise ValueError("time must be finite")
        for name, value in zip(
            ("theta", "theta_dot", "wheel_position", "wheel_velocity"),
            state.as_tuple(),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        del parameters

        if self._previous_time is None:
            timestep = 0.0
        else:
            timestep = time - self._previous_time
            if timestep <= 0.0:
                raise ValueError("time must increase between controller calls")
        self._previous_time = time

        # Outer position loop: positive position error produces a positive
        # angle reference under the plant's actuator sign convention.
        position_error = state.wheel_position - self.desired_position
        angle_reference = (
            self.desired_angle
            + self.position_kp * position_error
            + self.position_kd * state.wheel_velocity
        )
        if self.angle_reference_limit is not None:
            angle_reference = max(
                -self.angle_reference_limit,
                min(self.angle_reference_limit, angle_reference),
            )

        # Inner angle loop. Use the measured error derivative to avoid a
        # second numerical derivative of the raw sensor signal.
        angle_error = state.theta - angle_reference
        if self._previous_angle_error is None or timestep == 0.0:
            angle_derivative = 0.0
        else:
            angle_derivative = (
                angle_error - self._previous_angle_error
            ) / timestep
        self._previous_angle_error = angle_error

        candidate_integral = self._angle_integral + angle_error * timestep
        raw_torque = (
            self.angle_kp * angle_error
            + self.angle_ki * candidate_integral
            + self.angle_kd * angle_derivative
        )

        # Conditional integration is the anti-windup mechanism.
        if self._integral_update_allowed(raw_torque, angle_error):
            self._angle_integral = candidate_integral

        torque = (
            self.angle_kp * angle_error
            + self.angle_ki * self._angle_integral
            + self.angle_kd * angle_derivative
        )
        return self._saturate(torque)


__all__ = ["PIDController"]
