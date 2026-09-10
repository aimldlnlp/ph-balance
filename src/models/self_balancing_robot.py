"""Nonlinear equations of motion for the planar self-balancing robot.

The model uses generalized coordinates ``q = [s, theta]`` where ``s`` is
wheel position and ``theta = 0`` is upright. The public state keeps the
project-required ordering ``[theta, theta_dot, wheel_position,
wheel_velocity]``.

The equations are

.. math::

   M(q) q_ddot + C(q, q_dot) q_dot + g(q) + D q_dot = B tau

with

.. math::

   M(q) = [[M_t, m_b l cos(theta)],
           [m_b l cos(theta), I_b + m_b l^2]],

   C(q, q_dot) q_dot = [-m_b l sin(theta) theta_dot^2, 0]^T,

   g(q) = [0, -m_b g l sin(theta)]^T,
   D q_dot = [c_s s_dot, c_theta theta_dot]^T,
   B = [1/r, -1]^T.

``tau`` is the equivalent total torque from both wheels. The factor of two
for two wheels is already represented in the effective translational mass;
the actuator convention is explicitly kept in ``B``.
"""

from math import cos, isfinite, sin

from src.models.parameters import RobotParameters
from src.models.state import RobotState

Matrix2 = tuple[tuple[float, float], tuple[float, float]]
Vector2 = tuple[float, float]


def _validate_finite(name: str, value: float) -> None:
    """Reject non-finite scalar inputs before they reach the dynamics."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_parameters(parameters: RobotParameters) -> None:
    """Validate the physical values required by the model."""

    positive = (
        ("body_mass", parameters.body_mass),
        ("body_inertia", parameters.body_inertia),
        ("center_of_mass_length", parameters.center_of_mass_length),
        ("wheel_radius", parameters.wheel_radius),
        ("gravity", parameters.gravity),
    )
    nonnegative = (
        ("wheel_mass_per_wheel", parameters.wheel_mass_per_wheel),
        ("wheel_inertia_per_wheel", parameters.wheel_inertia_per_wheel),
        ("translational_friction", parameters.translational_friction),
        ("rotational_friction", parameters.rotational_friction),
    )

    for name, value in positive:
        _validate_finite(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    for name, value in nonnegative:
        _validate_finite(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")


def _validate_state(state: RobotState) -> None:
    """Validate all four components of the project state vector."""

    for name, value in zip(
        ("theta", "theta_dot", "wheel_position", "wheel_velocity"),
        state.as_tuple(),
    ):
        _validate_finite(name, value)


def _mass_matrix_unchecked(theta: float, parameters: RobotParameters) -> Matrix2:
    """Build the two-degree-of-freedom mass matrix after validation."""

    body_mass = parameters.body_mass
    length = parameters.center_of_mass_length
    effective_mass = (
        body_mass
        + 2.0 * parameters.wheel_mass_per_wheel
        + 2.0 * parameters.wheel_inertia_per_wheel / parameters.wheel_radius**2
    )
    body_pitch_inertia = parameters.body_inertia + body_mass * length**2
    coupling = body_mass * length * cos(theta)

    # Equation (1): M(q) = [[M_t, m_b*l*cos(theta)],
    #                       [m_b*l*cos(theta), I_b + m_b*l^2]].
    return (
        (effective_mass, coupling),
        (coupling, body_pitch_inertia),
    )


def mass_matrix(theta: float, parameters: RobotParameters) -> Matrix2:
    """Return the validated ``2 x 2`` configuration-dependent mass matrix."""

    _validate_parameters(parameters)
    _validate_finite("theta", theta)
    return _mass_matrix_unchecked(theta, parameters)


def _solve_mass_matrix(matrix: Matrix2, rhs: Vector2) -> Vector2:
    """Solve a validated ``2 x 2`` mass-matrix equation."""

    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("mass matrix must have shape (2, 2)")
    if len(rhs) != 2:
        raise ValueError("right-hand side must have dimension 2")

    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if a <= 0.0 or determinant <= 0.0 or not isfinite(determinant):
        raise ValueError("mass matrix must be positive definite")

    # Explicit 2 x 2 solve: M(q) q_ddot = rhs.
    return (
        (d * rhs[0] - b * rhs[1]) / determinant,
        (a * rhs[1] - c * rhs[0]) / determinant,
    )


def equations_of_motion(
    state: RobotState,
    motor_torque: float,
    parameters: RobotParameters,
    external_torque: float = 0.0,
) -> Vector2:
    """Return ``(wheel_acceleration, theta_acceleration)``.

    The returned accelerations solve the full nonlinear equations without a
    small-angle approximation.
    """

    _validate_parameters(parameters)
    _validate_state(state)
    _validate_finite("motor_torque", motor_torque)
    _validate_finite("external_torque", external_torque)

    theta = state.theta
    theta_dot = state.theta_dot
    wheel_velocity = state.wheel_velocity
    body_mass = parameters.body_mass
    length = parameters.center_of_mass_length

    matrix = _mass_matrix_unchecked(theta, parameters)

    # Equation (2): C(q, q_dot) q_dot =
    #                [-m_b*l*sin(theta)*theta_dot^2, 0].
    coriolis = (
        -body_mass * length * sin(theta) * theta_dot**2,
        0.0,
    )

    # Equation (3): g(q) = [0, -m_b*g*l*sin(theta)].
    gravity = (
        0.0,
        -body_mass * parameters.gravity * length * sin(theta),
    )

    # Equation (4): D q_dot = [c_s*s_dot, c_theta*theta_dot].
    damping = (
        parameters.translational_friction * wheel_velocity,
        parameters.rotational_friction * theta_dot,
    )

    # Equation (5): B*tau = [tau/r, -tau].
    actuation = (
        motor_torque / parameters.wheel_radius,
        -motor_torque,
    )

    # The disturbance is a generalized torque applied directly to the body
    # pitch coordinate. It is separate from the commanded motor torque so
    # disturbance-rejection experiments can be logged independently.
    disturbance = (0.0, external_torque)

    # Rearranged equation (6): M(q) q_ddot = B*tau + d - C*q_dot - g - D*q_dot.
    rhs: Vector2 = (
        actuation[0] - coriolis[0] - gravity[0] - damping[0],
        actuation[1]
        + disturbance[1]
        - coriolis[1]
        - gravity[1]
        - damping[1],
    )

    return _solve_mass_matrix(matrix, rhs)


def state_derivative(
    state: RobotState,
    motor_torque: float,
    parameters: RobotParameters,
    external_torque: float = 0.0,
) -> RobotState:
    """Return the derivative of the project state representation.

    Equation (7) maps generalized accelerations back to the required state
    order: ``[theta_dot, theta_ddot, wheel_velocity, wheel_acceleration]``.
    """

    wheel_acceleration, theta_acceleration = equations_of_motion(
        state,
        motor_torque,
        parameters,
        external_torque,
    )
    return RobotState(
        theta=state.theta_dot,
        theta_dot=theta_acceleration,
        wheel_position=state.wheel_velocity,
        wheel_velocity=wheel_acceleration,
    )


__all__ = [
    "equations_of_motion",
    "mass_matrix",
    "state_derivative",
]
