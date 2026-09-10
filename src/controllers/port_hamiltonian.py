r"""Port-Hamiltonian representation and energy-based controller.

The canonical system state is ``x = [s, theta, p_s, p_theta]``. Its
representation is

.. math::

   \dot{x} = (J - R) \nabla H(x) + G u,

where ``J`` is skew-symmetric energy exchange, ``R`` is positive
semidefinite dissipation, and ``G`` maps motor torque into generalized force.
The plant representation uses the physical gravitational Hamiltonian; the
controller separately uses an upright-relative shaped Hamiltonian.

The controller uses the requested energy-shaping and damping-injection law

.. math::

   u = -K_e (H_c - H_d) - K_d y,
   \qquad y = B^T \dot{q} = \dot{s}/r - \dot{theta}.

The original ``q_dot`` damping term is projected onto the scalar actuator
power port because this robot has one motor-torque input.
"""

from dataclasses import dataclass
from math import isfinite, sin
from typing import Optional

from src.models.hamiltonian import physical_energy_gradient, total_energy
from src.models.parameters import RobotParameters
from src.models.self_balancing_robot import mass_matrix
from src.models.state import RobotState

Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
Vector4 = tuple[float, float, float, float]
StateFeedbackGains = tuple[float, float, float, float]


def structure_matrix() -> Matrix4:
    """Return the canonical skew-symmetric interconnection matrix ``J``."""

    return (
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
    )


def dissipation_matrix(parameters: RobotParameters) -> Matrix4:
    """Return the viscous dissipation matrix ``R``."""

    # Reuse model validation, including the physical parameter checks.
    mass_matrix(0.0, parameters)
    return (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, parameters.translational_friction, 0.0),
        (0.0, 0.0, 0.0, parameters.rotational_friction),
    )


def input_matrix(parameters: RobotParameters) -> Vector4:
    """Return the motor-torque mapping ``G`` for the canonical state."""

    mass_matrix(0.0, parameters)
    return (0.0, 0.0, 1.0 / parameters.wheel_radius, -1.0)


def _matrix_vector_product(matrix: Matrix4, vector: Vector4) -> Vector4:
    """Multiply a validated four-by-four matrix by a four-vector."""

    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("port-Hamiltonian matrix must have shape (4, 4)")
    if len(vector) != 4:
        raise ValueError("port-Hamiltonian vector must have dimension 4")

    return (
        sum(matrix[0][index] * vector[index] for index in range(4)),
        sum(matrix[1][index] * vector[index] for index in range(4)),
        sum(matrix[2][index] * vector[index] for index in range(4)),
        sum(matrix[3][index] * vector[index] for index in range(4)),
    )


def port_hamiltonian_derivative(
    state: RobotState,
    motor_torque: float,
    parameters: RobotParameters,
    external_torque: float = 0.0,
) -> Vector4:
    """Return ``(J-R) grad(H) + G*u + d`` in canonical coordinates.

    The returned order is ``[s_dot, theta_dot, p_s_dot, p_theta_dot]``.
    """

    if not isfinite(motor_torque):
        raise ValueError("motor_torque must be finite")
    if not isfinite(external_torque):
        raise ValueError("external_torque must be finite")

    # The plant representation uses the physical gravitational potential.
    # The shaped control Hamiltonian remains separate in the controller.
    gradient = physical_energy_gradient(state, parameters)
    structure = _matrix_vector_product(structure_matrix(), gradient)
    dissipation = _matrix_vector_product(
        dissipation_matrix(parameters),
        gradient,
    )
    actuation = input_matrix(parameters)

    # Equation: x_dot = J*grad(H) - R*grad(H) + G*u + d.
    return (
        structure[0] - dissipation[0] + actuation[0] * motor_torque,
        structure[1] - dissipation[1] + actuation[1] * motor_torque,
        structure[2] - dissipation[2] + actuation[2] * motor_torque,
        structure[3]
        - dissipation[3]
        + actuation[3] * motor_torque
        + external_torque,
    )


@dataclass(frozen=True)
class PortHamiltonianController:
    """Configurable energy-shaping and damping-injection reference law.

    This scalar energy law is intentionally kept as the research reference
    controller. It is not guaranteed to stabilize the inverted plant because
    energy alone does not encode the lean direction.
    """

    energy_gain: float
    damping_gain: float
    desired_energy: float = 0.0
    torque_limit: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate controller parameters without tuning them."""

        for name, value in (
            ("energy_gain", self.energy_gain),
            ("damping_gain", self.damping_gain),
            ("desired_energy", self.desired_energy),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.energy_gain < 0.0 or self.damping_gain < 0.0:
            raise ValueError("controller gains must be non-negative")
        if self.torque_limit is not None:
            if not isfinite(self.torque_limit) or self.torque_limit <= 0.0:
                raise ValueError("torque_limit must be positive and finite")

    def compute_torque(
        self,
        time: float,
        state: RobotState,
        parameters: RobotParameters,
    ) -> float:
        """Return motor torque from energy error and port damping.

        ``time`` is accepted to satisfy the common controller interface; the
        current law is memoryless and does not use it.
        """

        if not isfinite(time):
            raise ValueError("time must be finite")

        energy_error = total_energy(state, parameters) - self.desired_energy
        actuator_velocity = (
            state.wheel_velocity / parameters.wheel_radius - state.theta_dot
        )
        torque = (
            -self.energy_gain * energy_error
            - self.damping_gain * actuator_velocity
        )

        if self.torque_limit is not None:
            torque = max(-self.torque_limit, min(self.torque_limit, torque))
        return torque


@dataclass(frozen=True)
class PortHamiltonianV2Controller:
    """Locally stabilizing extension of the PH-V1 reference law.

    The controller combines energy regulation, directional potential shaping,
    and port damping:

    .. math::

       \\tau = -K_e(H_c-H_d)y + K_\\theta\\sin(\\theta) - K_d y.

    The positive sign on the angle term follows the project's motor-torque
    convention ``B = [1/r, -1]``. A positive motor torque creates negative
    pitch acceleration near upright, so it restores a positive lean. Damping
    is applied at the actuator power port ``y`` to preserve the
    Port-Hamiltonian dissipation interpretation.

    This controller is intentionally separate from :class:`PortHamiltonianController`,
    which remains the PH-V1 scalar-energy baseline.
    """

    energy_gain: float
    angle_gain: float
    damping_gain: float
    desired_energy: float = 0.0
    torque_limit: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate finite gains and the optional motor torque limit."""

        for name, value in (
            ("energy_gain", self.energy_gain),
            ("angle_gain", self.angle_gain),
            ("damping_gain", self.damping_gain),
            ("desired_energy", self.desired_energy),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.energy_gain < 0.0
            or self.angle_gain < 0.0
            or self.damping_gain < 0.0
        ):
            raise ValueError("controller gains must be non-negative")
        if self.torque_limit is not None:
            if not isfinite(self.torque_limit) or self.torque_limit <= 0.0:
                raise ValueError("torque_limit must be positive and finite")

    def compute_torque(
        self,
        time: float,
        state: RobotState,
        parameters: RobotParameters,
    ) -> float:
        """Return the PH-V2 motor torque for the current robot state."""

        if not isfinite(time):
            raise ValueError("time must be finite")

        energy_error = total_energy(state, parameters) - self.desired_energy
        actuator_velocity = (
            state.wheel_velocity / parameters.wheel_radius - state.theta_dot
        )
        torque = (
            -self.energy_gain * energy_error * actuator_velocity
            + self.angle_gain * sin(state.theta)
            - self.damping_gain * actuator_velocity
        )

        if self.torque_limit is not None:
            torque = max(-self.torque_limit, min(self.torque_limit, torque))
        return torque


@dataclass(frozen=True)
class PortHamiltonianV3Controller:
    r"""PH-V3 reference law with angle and wheel-position feedback.

    PH-V3 extends PH-V2 with wheel translation feedback:

    .. math::

       \tau = -K_e(H_c-H_d)y + K_\theta\sin(\theta)
              + K_s s + K_v\dot{s}.

    The implicit regulation target is ``s = 0`` and ``dot(s) = 0``.  The
    positive motor-torque signs on the position and velocity terms are
    intentional for this model's input convention ``B = [1/r, -1]``.  They
    are the locally stabilizing signs found from the linearized motor-input
    model; copying generalized-force gradient signs directly would reverse
    the motor mapping.  This is a practical local reference law, not a claim
    of a complete IDA-PBC stability proof.

    The energy term uses the upright-relative shaped Hamiltonian supplied by
    :func:`src.models.hamiltonian.total_energy`, while the plant itself uses
    the physical gravitational Hamiltonian.
    """

    energy_gain: float
    angle_gain: float
    position_gain: float
    damping_gain: float
    desired_energy: float = 0.0
    torque_limit: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate finite gains and the optional motor torque limit."""

        for name, value in (
            ("energy_gain", self.energy_gain),
            ("angle_gain", self.angle_gain),
            ("position_gain", self.position_gain),
            ("damping_gain", self.damping_gain),
            ("desired_energy", self.desired_energy),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.energy_gain < 0.0
            or self.angle_gain < 0.0
            or self.position_gain < 0.0
            or self.damping_gain < 0.0
        ):
            raise ValueError("controller gains must be non-negative")
        if self.torque_limit is not None:
            if not isfinite(self.torque_limit) or self.torque_limit <= 0.0:
                raise ValueError("torque_limit must be positive and finite")

    def compute_torque(
        self,
        time: float,
        state: RobotState,
        parameters: RobotParameters,
    ) -> float:
        """Return the PH-V3 motor torque for the current robot state."""

        if not isfinite(time):
            raise ValueError("time must be finite")

        energy_error = total_energy(state, parameters) - self.desired_energy
        actuator_velocity = (
            state.wheel_velocity / parameters.wheel_radius - state.theta_dot
        )

        # Energy shaping at the actuator power port: -Ke*(H-Hd)*y.
        energy_term = -self.energy_gain * energy_error * actuator_velocity
        # Directional upright-angle shaping for B = [1/r, -1].
        angle_term = self.angle_gain * sin(state.theta)
        # Local wheel-position regulation around s = 0.
        position_term = self.position_gain * state.wheel_position
        # Motor-convention velocity feedback used to damp the translational mode.
        damping_term = self.damping_gain * state.wheel_velocity
        torque = energy_term + angle_term + position_term + damping_term

        if self.torque_limit is not None:
            torque = max(-self.torque_limit, min(self.torque_limit, torque))
        return torque


@dataclass(frozen=True)
class HybridPortHamiltonianController:
    r"""Combine port-Hamiltonian shaping with local state feedback.

    The hybrid law is

    .. math::

       u = -K_e(H_c-H_d)y - K_d y - Kx,

    where ``x`` follows the project state order
    ``[theta, theta_dot, wheel_position, wheel_velocity]`` and
    ``y = wheel_velocity / r - theta_dot`` is the actuator power-port
    velocity. ``stabilizer_gains`` is the row vector ``K``; its signs are
    intentionally not restricted because the plant input mapping determines
    the restoring signs. A gain row computed by LQR can therefore be passed
    directly without rewriting its signs.

    The PH and state-feedback terms are summed before one optional torque
    saturation. This keeps the two contributions visible for validation and
    avoids independently saturating components before they interact.
    """

    energy_gain: float
    damping_gain: float
    stabilizer_gains: StateFeedbackGains
    desired_energy: float = 0.0
    torque_limit: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate hybrid gains, state-feedback dimensions, and limits."""

        for name, value in (
            ("energy_gain", self.energy_gain),
            ("damping_gain", self.damping_gain),
            ("desired_energy", self.desired_energy),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.energy_gain < 0.0 or self.damping_gain < 0.0:
            raise ValueError("energy and damping gains must be non-negative")
        if len(self.stabilizer_gains) != 4:
            raise ValueError("stabilizer_gains must have four elements")
        if not all(isfinite(value) for value in self.stabilizer_gains):
            raise ValueError("stabilizer_gains must be finite")
        if self.torque_limit is not None:
            if not isfinite(self.torque_limit) or self.torque_limit <= 0.0:
                raise ValueError("torque_limit must be positive and finite")

    def compute_torque(
        self,
        time: float,
        state: RobotState,
        parameters: RobotParameters,
    ) -> float:
        """Return the summed PH and local state-feedback motor torque."""

        if not isfinite(time):
            raise ValueError("time must be finite")

        energy_error = total_energy(state, parameters) - self.desired_energy
        actuator_velocity = (
            state.wheel_velocity / parameters.wheel_radius - state.theta_dot
        )
        state_vector = state.as_tuple()
        energy_term = -self.energy_gain * energy_error * actuator_velocity
        damping_term = -self.damping_gain * actuator_velocity
        stabilizer_term = -sum(
            gain * value
            for gain, value in zip(self.stabilizer_gains, state_vector)
        )
        torque = energy_term + damping_term + stabilizer_term

        if self.torque_limit is not None:
            torque = max(-self.torque_limit, min(self.torque_limit, torque))
        return torque


__all__ = [
    "HybridPortHamiltonianController",
    "PortHamiltonianController",
    "PortHamiltonianV2Controller",
    "PortHamiltonianV3Controller",
    "dissipation_matrix",
    "input_matrix",
    "port_hamiltonian_derivative",
    "structure_matrix",
]
