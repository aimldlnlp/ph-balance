"""Run the fixed PH-Balance controller comparison experiments.

Run from the repository root with::

    python -m experiments.run_experiments

Each scenario writes its comparison plot, controller trajectories, metrics,
and a short Markdown summary under ``results/experiments``. The controller
receives nominal parameters in the uncertainty experiment; the plant receives
the perturbed parameters. This keeps the robustness test meaningful.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import PIDController
from src.controllers.port_hamiltonian import PortHamiltonianController
from src.controllers.protocols import Controller
from src.models.parameters import RobotParameters
from src.models.self_balancing_robot import state_derivative
from src.models.state import RobotState
from src.simulation.simulator import Disturbance, SimulationConfig, SimulationResult, Simulator
from src.visualization.plots import plot_controller_comparison

from experiments.metrics import ExperimentMetrics, compute_metrics


ControllerBuilder = Callable[[], Controller]

PID_GAINS = {
    "angle_kp": 30.0,
    "angle_ki": 0.5,
    "angle_kd": 3.0,
    "torque_limit": 3.0,
    "angle_reference_limit": 0.3,
}
PH_REFERENCE_GAINS = {
    "energy_gain": 0.1,
    "damping_gain": 1.0,
    "torque_limit": 3.0,
}
MAX_VALID_ANGLE = float(np.deg2rad(85.0))


@dataclass(frozen=True)
class ExperimentSpec:
    """Fixed initial condition, plant, and disturbance for one scenario."""

    name: str
    initial_state: RobotState
    duration: float
    timestep: float
    plant_parameters: RobotParameters
    max_abs_angle: float | None = MAX_VALID_ANGLE
    disturbance: Disturbance | None = None
    disturbance_time: float | None = None


def nominal_parameters() -> RobotParameters:
    """Return the single nominal parameter set used by all controllers."""

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


def controller_builders() -> dict[str, ControllerBuilder]:
    """Return fresh, fixed controller configurations for every run."""

    return {
        "PID": lambda: PIDController(**PID_GAINS),
        "Port-Hamiltonian": lambda: PortHamiltonianController(
            **PH_REFERENCE_GAINS
        ),
    }


def disturbance_torque(time: float) -> float:
    """Apply a deterministic 1 N m body-torque pulse beginning at 5 s."""

    return 1.0 if 5.0 <= time < 5.2 else 0.0


def experiment_specs() -> tuple[ExperimentSpec, ...]:
    """Build the three scenarios described in ``EXPERIMENT_PLAN.md``."""

    nominal = nominal_parameters()
    initial_lean = RobotState(float(np.deg2rad(15.0)), 0.0, 0.0, 0.0)
    upright = RobotState(0.0, 0.0, 0.0, 0.0)
    uncertain = replace(
        nominal,
        body_mass=nominal.body_mass * 1.25,
        translational_friction=nominal.translational_friction * 2.0,
        rotational_friction=nominal.rotational_friction * 2.0,
    )

    return (
        ExperimentSpec(
            name="nominal_stabilization",
            initial_state=initial_lean,
            duration=4.0,
            timestep=0.02,
            plant_parameters=nominal,
        ),
        ExperimentSpec(
            name="external_disturbance",
            initial_state=upright,
            duration=8.0,
            timestep=0.02,
            plant_parameters=nominal,
            disturbance=disturbance_torque,
            disturbance_time=5.0,
        ),
        ExperimentSpec(
            name="parameter_uncertainty",
            initial_state=initial_lean,
            duration=4.0,
            timestep=0.02,
            plant_parameters=uncertain,
        ),
    )


def _write_metrics(path: Path, metrics: list[ExperimentMetrics]) -> None:
    """Write one row per controller in stable column order."""

    rows = [metric.as_dict() for metric in metrics]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _display_value(value: object) -> str:
    """Format metric values for the Markdown summary."""

    if isinstance(value, float):
        return "-" if not np.isfinite(value) else f"{value:.3f}"
    if isinstance(value, str) and not value:
        return "-"
    return str(value)


def _write_markdown_summary(path: Path, metrics: list[ExperimentMetrics]) -> None:
    """Write a compact human-readable summary table."""

    columns = (
        ("Controller", "controller"),
        ("Duration (s)", "simulated_duration_s"),
        ("Settling (s)", "settling_time_s"),
        ("RMSE (deg)", "rmse_angle_deg"),
        ("Peak (deg)", "peak_angle_deg"),
        ("Recovery (s)", "recovery_time_s"),
        ("Peak torque (N m)", "peak_torque_nm"),
        ("Mean abs torque (N m)", "mean_abs_torque_nm"),
        ("Energy deviation RMS (J)", "energy_deviation_rms_j"),
        ("Success", "success"),
        ("Termination time (s)", "termination_time_s"),
        ("Fallover penalty", "fallover_penalty"),
        ("Termination", "termination_reason"),
    )
    lines = [
        "| " + " | ".join(header for header, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for metric in metrics:
        values = metric.as_dict()
        lines.append(
            "| "
            + " | ".join(
                _display_value(values[key]) for _, key in columns
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_configuration(
    path: Path,
    spec: ExperimentSpec,
    controller_parameters: RobotParameters,
) -> None:
    """Record the fixed scenario settings beside generated results."""

    payload = {
        "name": spec.name,
        "duration_s": spec.duration,
        "timestep_s": spec.timestep,
        "max_abs_angle_deg": (
            None
            if spec.max_abs_angle is None
            else float(np.rad2deg(spec.max_abs_angle))
        ),
        "initial_state": asdict(spec.initial_state),
        "plant_parameters": asdict(spec.plant_parameters),
        "controller_parameters": asdict(controller_parameters),
        "disturbance": {
            "type": "body_torque_pulse",
            "start_time_s": spec.disturbance_time,
            "amplitude_nm": 1.0,
            "duration_s": 0.2,
        }
        if spec.disturbance is not None
        else None,
        "controllers": {
            "PID": {
                "angle_kp": PID_GAINS["angle_kp"],
                "angle_ki": PID_GAINS["angle_ki"],
                "angle_kd": PID_GAINS["angle_kd"],
                "torque_limit_nm": PID_GAINS["torque_limit"],
            },
            "Port-Hamiltonian": {
                "energy_gain": PH_REFERENCE_GAINS["energy_gain"],
                "damping_gain": PH_REFERENCE_GAINS["damping_gain"],
                "torque_limit_nm": PH_REFERENCE_GAINS["torque_limit"],
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_experiment(
    spec: ExperimentSpec,
    output_root: Path,
    controller_parameters: RobotParameters,
) -> list[ExperimentMetrics]:
    """Run both controllers and write all artifacts for one scenario."""

    output_directory = output_root / spec.name
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_configuration(
        output_directory / "configuration.json",
        spec,
        controller_parameters,
    )

    results: dict[str, SimulationResult] = {}
    metrics: list[ExperimentMetrics] = []
    for controller_name, build_controller in controller_builders().items():
        result = Simulator(
            model=state_derivative,
            controller=build_controller(),
            parameters=spec.plant_parameters,
            controller_parameters=controller_parameters,
            config=SimulationConfig(
                duration=spec.duration,
                timestep=spec.timestep,
                max_abs_angle=spec.max_abs_angle,
            ),
            disturbance=spec.disturbance,
        ).run(spec.initial_state)
        results[controller_name] = result
        result.export_csv(output_directory / f"{controller_name.lower().replace('-', '_')}_trajectory.csv")
        metrics.append(
            compute_metrics(
                result,
                experiment=spec.name,
                controller=controller_name,
                disturbance_time=spec.disturbance_time,
            )
        )

    plot = plot_controller_comparison(
        results,
        output_directory / "controller_comparison.png",
        title=spec.name.replace("_", " ").title(),
        disturbance_time=spec.disturbance_time,
    )
    plt.close(plot)

    _write_metrics(output_directory / "metrics.csv", metrics)
    _write_markdown_summary(output_directory / "summary.md", metrics)
    return metrics


def _write_root_summary(path: Path, metrics: list[ExperimentMetrics]) -> None:
    """Write the cross-experiment comparison table."""

    columns = (
        ("Experiment", "experiment"),
        ("Controller", "controller"),
        ("Duration (s)", "simulated_duration_s"),
        ("Settling (s)", "settling_time_s"),
        ("RMSE (deg)", "rmse_angle_deg"),
        ("Max deviation (deg)", "max_deviation_deg"),
        ("Recovery (s)", "recovery_time_s"),
        ("Peak torque (N m)", "peak_torque_nm"),
        ("Mean abs torque (N m)", "mean_abs_torque_nm"),
        ("Energy deviation RMS (J)", "energy_deviation_rms_j"),
        ("Success", "success"),
        ("Termination time (s)", "termination_time_s"),
        ("Fallover penalty", "fallover_penalty"),
        ("Termination", "termination_reason"),
    )
    with path.open("w", encoding="utf-8") as output:
        output.write("| " + " | ".join(header for header, _ in columns) + " |\n")
        output.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for metric in metrics:
            values = metric.as_dict()
            output.write(
                "| "
                + " | ".join(
                    _display_value(values[key]) for _, key in columns
                )
                + " |\n"
            )


def main() -> None:
    """Run every configured scenario and write aggregate summaries."""

    output_root = ROOT / "results" / "experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    controller_parameters = nominal_parameters()
    all_metrics: list[ExperimentMetrics] = []
    for spec in experiment_specs():
        all_metrics.extend(
            run_experiment(spec, output_root, controller_parameters)
        )

    _write_metrics(output_root / "metrics.csv", all_metrics)
    _write_root_summary(output_root / "summary.md", all_metrics)
    print(f"Wrote {len(all_metrics)} controller results to {output_root}")


if __name__ == "__main__":
    main()
