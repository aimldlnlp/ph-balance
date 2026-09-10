r"""Physical and control Hamiltonians for the self-balancing robot.

The project state stores velocities, while the Hamiltonian uses canonical
momentum. For ``q = [s, theta]`` and ``p = M(q) q_dot``:

.. math::

   H(q, p) = T(q, p) + V(q)
             = \frac{1}{2} p^T M(q)^{-1} p + V(q).

The controller uses an upright-relative shaped potential. Therefore
``potential_energy`` is the control potential

.. math::

   V(theta) = m_b g l (1 - cos(theta)).

This is the control Hamiltonian used for energy shaping. It is not the raw
plant gravitational potential. ``physical_potential_energy`` and
``physical_total_energy`` provide the latter and are used by the physical
simulation and the port-Hamiltonian plant representation.

``energy_gradient`` returns the shaped-control canonical gradient in the order
``[dH/ds, dH/dtheta, dH/dp_s, dH/dp_theta]``.
"""

from math import cos, isfinite, sin

from src.models.parameters import RobotParameters
from src.models.self_balancing_robot import mass_matrix
from src.models.state import RobotState

CanonicalGradient = tuple[float, float, float, float]


def _validate_state(state: RobotState) -> None:
    """Reject non-finite state values before evaluating energy."""

    for name, value in zip(
        ("theta", "theta_dot", "wheel_position", "wheel_velocity"),
        state.as_tuple(),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")


def _canonical_momentum(
    state: RobotState,
    parameters: RobotParameters,
) -> tuple[float, float]:
    """Return ``p = M(q) q_dot`` in ``[p_s, p_theta]`` order."""

    matrix = mass_matrix(state.theta, parameters)
    q_dot = (state.wheel_velocity, state.theta_dot)
    return (
        matrix[0][0] * q_dot[0] + matrix[0][1] * q_dot[1],
        matrix[1][0] * q_dot[0] + matrix[1][1] * q_dot[1],
    )


def kinetic_energy(state: RobotState, parameters: RobotParameters) -> float:
    """Return ``T(q,p) = 1/2 p^T M(q)^-1 p`` in joules."""

    _validate_state(state)
    momentum = _canonical_momentum(state, parameters)
    q_dot = (state.wheel_velocity, state.theta_dot)

    # Since M^-1 p = q_dot, this is the same Hamiltonian expression without
    # explicitly inverting M.
    return 0.5 * (momentum[0] * q_dot[0] + momentum[1] * q_dot[1])


def potential_energy(state: RobotState, parameters: RobotParameters) -> float:
    """Return upright-relative control-shaped potential energy in joules.

    The reference is chosen so that upright rest has zero potential energy.
    """

    _validate_state(state)
    # Call mass_matrix to reuse the model's parameter and angle validation.
    mass_matrix(state.theta, parameters)
    return (
        parameters.body_mass
        * parameters.gravity
        * parameters.center_of_mass_length
        * (1.0 - cos(state.theta))
    )


def total_energy(state: RobotState, parameters: RobotParameters) -> float:
    """Return the shaped control Hamiltonian ``H_c = T + V_c``."""

    return kinetic_energy(state, parameters) + potential_energy(state, parameters)


def physical_potential_energy(
    state: RobotState,
    parameters: RobotParameters,
) -> float:
    """Return the plant gravitational potential ``m g l cos(theta)``."""

    _validate_state(state)
    mass_matrix(state.theta, parameters)
    return (
        parameters.body_mass
        * parameters.gravity
        * parameters.center_of_mass_length
        * cos(state.theta)
    )


def physical_total_energy(
    state: RobotState,
    parameters: RobotParameters,
) -> float:
    """Return the plant's conserved mechanical Hamiltonian."""

    return kinetic_energy(state, parameters) + physical_potential_energy(
        state,
        parameters,
    )


def _canonical_energy_gradient(
    state: RobotState,
    parameters: RobotParameters,
    potential_derivative: float,
) -> CanonicalGradient:
    """Build a canonical gradient for a supplied potential derivative."""

    _validate_state(state)
    mass_matrix(state.theta, parameters)

    theta = state.theta
    wheel_velocity = state.wheel_velocity
    theta_dot = state.theta_dot
    coupling = parameters.body_mass * parameters.center_of_mass_length

    # The kinetic contribution is evaluated at fixed canonical momentum.
    kinetic_derivative = coupling * sin(theta) * wheel_velocity * theta_dot
    return (
        0.0,
        kinetic_derivative + potential_derivative,
        wheel_velocity,
        theta_dot,
    )


def energy_gradient(
    state: RobotState,
    parameters: RobotParameters,
) -> CanonicalGradient:
    """Return the canonical gradient of the Hamiltonian.

    The derivative with respect to ``theta`` is evaluated at fixed canonical
    momentum, not fixed velocity. For the configuration-dependent mass matrix
    this contributes ``m_b*l*sin(theta)*s_dot*theta_dot``.
    """

    potential_derivative = (
        parameters.body_mass
        * parameters.gravity
        * parameters.center_of_mass_length
        * sin(state.theta)
    )
    return _canonical_energy_gradient(state, parameters, potential_derivative)


def physical_energy_gradient(
    state: RobotState,
    parameters: RobotParameters,
) -> CanonicalGradient:
    """Return the canonical gradient of the plant mechanical Hamiltonian."""

    potential_derivative = (
        -parameters.body_mass
        * parameters.gravity
        * parameters.center_of_mass_length
        * sin(state.theta)
    )
    return _canonical_energy_gradient(state, parameters, potential_derivative)


__all__ = [
    "energy_gradient",
    "kinetic_energy",
    "physical_energy_gradient",
    "physical_potential_energy",
    "physical_total_energy",
    "potential_energy",
    "total_energy",
]
