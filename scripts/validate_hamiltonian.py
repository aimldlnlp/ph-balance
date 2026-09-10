"""Small executable validation for the PH-Balance Hamiltonian."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    RobotParameters,
    RobotState,
    energy_gradient,
    physical_energy_gradient,
    physical_total_energy,
    potential_energy,
    total_energy,
)
from src.models.self_balancing_robot import mass_matrix  # noqa: E402


def make_parameters() -> RobotParameters:
    """Return the deterministic validation parameter set."""

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
    a, b = mass_matrix(theta, parameters)[0]
    c, d = mass_matrix(theta, parameters)[1]
    determinant = a * d - b * c
    return RobotState(
        theta=theta,
        theta_dot=(-c * p_s + a * p_theta) / determinant,
        wheel_position=wheel_position,
        wheel_velocity=(d * p_s - b * p_theta) / determinant,
    )


def main() -> None:
    """Run the three requested Hamiltonian checks."""

    parameters = make_parameters()
    equilibrium = RobotState(0.0, 0.0, 0.0, 0.0)
    displaced = RobotState(0.2, 0.0, 0.0, 0.0)
    assert abs(total_energy(equilibrium, parameters)) < 1.0e-12
    assert potential_energy(displaced, parameters) > potential_energy(
        equilibrium,
        parameters,
    )

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

    maximum_error = max(abs(expected - actual) for expected, actual in zip(analytic, numeric))
    assert maximum_error < 1.0e-6, maximum_error

    physical_analytic = physical_energy_gradient(state, parameters)
    physical_numeric: list[float] = []
    for index in range(4):
        plus = list(canonical_state)
        minus = list(canonical_state)
        plus[index] += step
        minus[index] -= step
        physical_numeric.append(
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

    physical_error = max(
        abs(expected - actual)
        for expected, actual in zip(physical_analytic, physical_numeric)
    )
    assert physical_error < 1.0e-6, physical_error
    print(
        "Hamiltonian validation passed; "
        f"shaped error={maximum_error:.3e}, "
        f"physical error={physical_error:.3e}"
    )


if __name__ == "__main__":
    main()
