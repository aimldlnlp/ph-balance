"""Generate the PH-Balance portfolio visualization artifacts."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.controllers import PIDController
from src.models import RobotParameters, RobotState, state_derivative
from src.simulation import SimulationConfig, Simulator
from src.visualization import (
    create_robot_animation,
    plot_angle_response,
    plot_control_effort,
    plot_energy_evolution,
)


def make_parameters() -> RobotParameters:
    """Return the deterministic nominal parameters for the demo artifact."""

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


def main() -> None:
    """Run one deterministic demo and save all requested visuals."""

    parameters = make_parameters()
    controller = PIDController(
        angle_kp=30.0,
        angle_ki=1.0,
        angle_kd=5.0,
        torque_limit=3.0,
        angle_reference_limit=0.3,
    )
    result = Simulator(
        model=state_derivative,
        controller=controller,
        parameters=parameters,
        config=SimulationConfig(duration=4.0, timestep=0.02),
    ).run(RobotState(np.deg2rad(15.0), 0.0, 0.0, 0.0))

    results_root = Path("results")
    plot_angle_response(result, results_root / "plots" / "angle_response.png")
    plot_energy_evolution(result, results_root / "plots" / "energy_evolution.png")
    plot_control_effort(result, results_root / "plots" / "control_effort.png")
    create_robot_animation(
        result,
        parameters,
        results_root / "animations" / "balance.gif",
    )
    print("Generated PH-Balance visualizations in results/")


if __name__ == "__main__":
    main()
