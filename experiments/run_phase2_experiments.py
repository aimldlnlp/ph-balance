"""Run reproducible controller comparisons for the Phase 2 experiments.

Run from the repository root with::

    python -m experiments.run_phase2_experiments

The runner writes raw trajectories, scalar metrics, comparison plots, sweep
plots, and a neutral summary under a configurable output directory. The
controller gains are fixed reference values; no optimization is performed
here.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import ExperimentMetrics, compute_metrics
from experiments.run_experiments import (
    PH_REFERENCE_GAINS,
    PID_GAINS,
    nominal_parameters,
)
from src.controllers import (
    Controller,
    HybridPortHamiltonianController,
    PortHamiltonianController,
    PortHamiltonianV2Controller,
    PortHamiltonianV3Controller,
)
from src.controllers.pid import PIDController
from src.models.parameters import RobotParameters
from src.models.self_balancing_robot import state_derivative
from src.models.state import RobotState
from src.simulation import Disturbance, SimulationConfig, SimulationResult, Simulator
from src.visualization.plots import (
    plot_controller_comparison,
    plot_sweep_metric,
)


ControllerBuilder = Callable[[], Controller]

PH_V2_REFERENCE_GAINS = {
    "energy_gain": 0.1,
    "angle_gain": 2.0,
    "damping_gain": 1.0,
    "desired_energy": 0.0,
    "torque_limit": 3.0,
}
PH_V3_REFERENCE_GAINS = {
    "energy_gain": 0.1,
    "angle_gain": 4.0,
    "position_gain": 0.1,
    "damping_gain": 0.01,
    "desired_energy": 0.0,
    "torque_limit": 3.0,
}
HYBRID_PH_REFERENCE_GAINS = {
    "energy_gain": 0.1,
    "damping_gain": 0.01,
    "stabilizer_gains": (
        -17.30522786,
        -4.96355526,
        -3.16227766,
        -5.90186191,
    ),
    "desired_energy": 0.0,
    "torque_limit": 3.0,
}
CONTROLLER_NAMES = ("PID", "PH-V1", "PH-V2", "PH-V3")
PHASE3_CONTROLLER_NAMES = (*CONTROLLER_NAMES, "Hybrid PH")
MAX_VALID_ANGLE = float(np.deg2rad(85.0))
DISTURBANCE_START_S = 5.0
DISTURBANCE_STOP_S = 5.2
TIMESTEP_S = 0.02


@dataclass(frozen=True)
class ExperimentSpec:
    """Initial condition, plant, disturbance, and sweep metadata."""

    group: str
    name: str
    initial_state: RobotState
    duration: float
    plant_parameters: RobotParameters
    max_abs_angle: float | None = MAX_VALID_ANGLE
    disturbance: Disturbance | None = None
    disturbance_time: float | None = None
    disturbance_amplitude_nm: float | None = None
    sweep_value: float | None = None


def controller_builders(
    include_hybrid: bool = False,
) -> dict[str, ControllerBuilder]:
    """Return fresh controller instances for the requested comparison set."""

    builders: dict[str, ControllerBuilder] = {
        "PID": lambda: PIDController(**PID_GAINS),
        "PH-V1": lambda: PortHamiltonianController(**PH_REFERENCE_GAINS),
        "PH-V2": lambda: PortHamiltonianV2Controller(
            **PH_V2_REFERENCE_GAINS
        ),
        "PH-V3": lambda: PortHamiltonianV3Controller(
            **PH_V3_REFERENCE_GAINS
        ),
    }
    if include_hybrid:
        builders["Hybrid PH"] = lambda: HybridPortHamiltonianController(
            **HYBRID_PH_REFERENCE_GAINS
        )
    return builders


def body_torque_pulse(amplitude_nm: float) -> Disturbance:
    """Create a deterministic body-torque pulse for the disturbance sweep."""

    if not np.isfinite(amplitude_nm) or amplitude_nm < 0.0:
        raise ValueError("amplitude_nm must be finite and non-negative")

    def pulse(time: float) -> float:
        return (
            float(amplitude_nm)
            if DISTURBANCE_START_S <= time < DISTURBANCE_STOP_S
            else 0.0
        )

    return pulse


def _initial_lean(angle_degrees: float) -> RobotState:
    """Build a stationary initial state from an angle in degrees."""

    return RobotState(float(np.deg2rad(angle_degrees)), 0.0, 0.0, 0.0)


def main_experiment_specs() -> tuple[ExperimentSpec, ...]:
    """Return the three required Phase 2 experiment specifications."""

    nominal = nominal_parameters()
    uncertain = RobotParameters(
        body_mass=nominal.body_mass * 1.25,
        wheel_mass_per_wheel=nominal.wheel_mass_per_wheel,
        body_inertia=nominal.body_inertia,
        wheel_inertia_per_wheel=nominal.wheel_inertia_per_wheel,
        center_of_mass_length=nominal.center_of_mass_length,
        wheel_radius=nominal.wheel_radius,
        gravity=nominal.gravity,
        translational_friction=nominal.translational_friction * 2.0,
        rotational_friction=nominal.rotational_friction * 2.0,
    )

    return (
        ExperimentSpec(
            group="main",
            name="stabilization",
            initial_state=_initial_lean(15.0),
            duration=4.0,
            plant_parameters=nominal,
        ),
        ExperimentSpec(
            group="main",
            name="disturbance_rejection",
            initial_state=_initial_lean(0.0),
            duration=8.0,
            plant_parameters=nominal,
            disturbance=body_torque_pulse(1.0),
            disturbance_time=DISTURBANCE_START_S,
            disturbance_amplitude_nm=1.0,
        ),
        ExperimentSpec(
            group="main",
            name="parameter_uncertainty",
            initial_state=_initial_lean(15.0),
            duration=4.0,
            plant_parameters=uncertain,
        ),
    )


def angle_sweep_specs() -> tuple[ExperimentSpec, ...]:
    """Return stabilization specifications for the requested angle sweep."""

    parameters = nominal_parameters()
    return tuple(
        ExperimentSpec(
            group="angle_sweep",
            name=f"theta_{angle:g}_deg",
            initial_state=_initial_lean(angle),
            duration=4.0,
            plant_parameters=parameters,
            sweep_value=float(angle),
        )
        for angle in (5.0, 15.0, 30.0, 45.0)
    )


def disturbance_sweep_specs() -> tuple[ExperimentSpec, ...]:
    """Return disturbance specifications for the requested torque sweep."""

    parameters = nominal_parameters()
    amplitudes = (0.2, 0.5, 1.0, 1.5, 2.0)
    return tuple(
        ExperimentSpec(
            group="disturbance_sweep",
            name=f"pulse_{amplitude:g}_nm".replace(".", "p"),
            initial_state=_initial_lean(0.0),
            duration=8.0,
            plant_parameters=parameters,
            disturbance=body_torque_pulse(amplitude),
            disturbance_time=DISTURBANCE_START_S,
            disturbance_amplitude_nm=float(amplitude),
            sweep_value=float(amplitude),
        )
        for amplitude in amplitudes
    )


def all_experiment_specs() -> tuple[ExperimentSpec, ...]:
    """Return main and sweep specifications in deterministic order."""

    return (
        *main_experiment_specs(),
        *angle_sweep_specs(),
        *disturbance_sweep_specs(),
    )


def _controller_file_name(controller: str) -> str:
    """Convert a controller label to a stable trajectory filename."""

    return controller.lower().replace("-", "_").replace(" ", "_")


def _write_metrics(path: Path, metrics: list[ExperimentMetrics]) -> None:
    """Write metric records to CSV using dataclass field order."""

    if not metrics:
        raise ValueError("metrics must not be empty")
    rows = [metric.as_dict() for metric in metrics]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _display_value(value: object) -> str:
    """Format scalar values for a neutral Markdown data table."""

    if isinstance(value, float):
        return "-" if not np.isfinite(value) else f"{value:.6f}"
    if isinstance(value, str) and not value:
        return "-"
    return str(value)


SUMMARY_COLUMNS = (
    ("Experiment", "experiment"),
    ("Controller", "controller"),
    ("Duration (s)", "simulated_duration_s"),
    ("Settling (s)", "settling_time_s"),
    ("RMSE angle (deg)", "rmse_angle_deg"),
    ("Overshoot (deg)", "overshoot_deg"),
    ("Max deviation (deg)", "max_deviation_deg"),
    ("Wheel drift (m)", "wheel_drift_m"),
    ("Recovery (s)", "recovery_time_s"),
    ("Energy deviation RMS (J)", "energy_deviation_rms_j"),
    ("Final energy deviation (J)", "final_energy_deviation_j"),
    ("Peak torque (N m)", "peak_torque_nm"),
    ("ZOH torque effort (N m s)", "integral_abs_torque_nm_s"),
    ("Success", "success"),
    ("Termination time (s)", "termination_time_s"),
    ("Fallover penalty", "fallover_penalty"),
    ("Termination", "termination_reason"),
)


def _markdown_table(metrics: Iterable[ExperimentMetrics]) -> str:
    """Return a Markdown table without adding result interpretation."""

    lines = [
        "| " + " | ".join(header for header, _ in SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for metric in metrics:
        values = metric.as_dict()
        lines.append(
            "| "
            + " | ".join(
                _display_value(values[key]) for _, key in SUMMARY_COLUMNS
            )
            + " |"
        )
    return "\n".join(lines)


def _write_spec_configuration(
    path: Path,
    spec: ExperimentSpec,
    controller_parameters: RobotParameters,
    include_hybrid: bool = False,
) -> None:
    """Write exact scenario and controller settings beside the data."""

    controllers = {
        "PID": PID_GAINS,
        "PH-V1": PH_REFERENCE_GAINS,
        "PH-V2": PH_V2_REFERENCE_GAINS,
        "PH-V3": PH_V3_REFERENCE_GAINS,
    }
    if include_hybrid:
        controllers["Hybrid PH"] = HYBRID_PH_REFERENCE_GAINS

    payload = {
        "group": spec.group,
        "name": spec.name,
        "duration_s": spec.duration,
        "timestep_s": TIMESTEP_S,
        "max_abs_angle_deg": (
            None
            if spec.max_abs_angle is None
            else float(np.rad2deg(spec.max_abs_angle))
        ),
        "initial_state": asdict(spec.initial_state),
        "plant_parameters": asdict(spec.plant_parameters),
        "controller_parameters": asdict(controller_parameters),
        "sweep_value": spec.sweep_value,
        "disturbance": (
            {
                "type": "body_torque_pulse",
                "start_time_s": DISTURBANCE_START_S,
                "stop_time_s": DISTURBANCE_STOP_S,
                "amplitude_nm": spec.disturbance_amplitude_nm,
            }
            if spec.disturbance is not None
            else None
        ),
        "controllers": controllers,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_spec(
    spec: ExperimentSpec,
    output_root: Path,
    controller_parameters: RobotParameters,
    include_hybrid: bool = False,
) -> list[ExperimentMetrics]:
    """Run the selected controllers for one specification and save artifacts."""

    output_directory = output_root / spec.group / spec.name
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_spec_configuration(
        output_directory / "configuration.json",
        spec,
        controller_parameters,
        include_hybrid=include_hybrid,
    )

    results: dict[str, SimulationResult] = {}
    metrics: list[ExperimentMetrics] = []
    for controller_name, build_controller in controller_builders(
        include_hybrid
    ).items():
        result = Simulator(
            model=state_derivative,
            controller=build_controller(),
            parameters=spec.plant_parameters,
            controller_parameters=controller_parameters,
            config=SimulationConfig(
                duration=spec.duration,
                timestep=TIMESTEP_S,
                max_abs_angle=spec.max_abs_angle,
            ),
            disturbance=spec.disturbance,
        ).run(spec.initial_state)
        results[controller_name] = result
        result.export_csv(
            output_directory / f"{_controller_file_name(controller_name)}_trajectory.csv"
        )
        metrics.append(
            compute_metrics(
                result,
                experiment=spec.name,
                controller=controller_name,
                disturbance_time=spec.disturbance_time,
            )
        )

    figure = plot_controller_comparison(
        results,
        output_directory / "controller_comparison.png",
        title=spec.name.replace("_", " ").title(),
        disturbance_time=spec.disturbance_time,
    )
    plt.close(figure)
    _write_metrics(output_directory / "metrics.csv", metrics)
    (output_directory / "summary.md").write_text(
        f"# {spec.name}\n\n{_markdown_table(metrics)}\n",
        encoding="utf-8",
    )
    return metrics


def _metrics_for_group(
    metrics: list[ExperimentMetrics],
    group_specs: tuple[ExperimentSpec, ...],
) -> list[ExperimentMetrics]:
    """Select metrics in the deterministic order of a sweep."""

    names = {spec.name for spec in group_specs}
    return [metric for metric in metrics if metric.experiment in names]


def _sweep_series(
    metrics: list[ExperimentMetrics],
    specs: tuple[ExperimentSpec, ...],
    metric_name: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build controller series for one sweep metric."""

    values = {
        spec.name: float(spec.sweep_value)
        for spec in specs
        if spec.sweep_value is not None
    }
    grouped: dict[str, list[tuple[float, float]]] = {}
    for metric in metrics:
        grouped.setdefault(metric.controller, []).append(
            (values[metric.experiment], float(getattr(metric, metric_name)))
        )

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for controller, points in grouped.items():
        points.sort(key=lambda point: point[0])
        series[controller] = (
            np.asarray([point[0] for point in points], dtype=float),
            np.asarray([point[1] for point in points], dtype=float),
        )
    return series


def _write_sweep_outputs(
    output_root: Path,
    metrics: list[ExperimentMetrics],
    specs: tuple[ExperimentSpec, ...],
    x_label: str,
    stem: str,
    plots: tuple[tuple[str, str, str], ...],
) -> None:
    """Write sweep CSV and requested metric plots."""

    _write_metrics(output_root / f"{stem}_metrics.csv", metrics)
    for metric_name, y_label, plot_name in plots:
        figure = plot_sweep_metric(
            _sweep_series(metrics, specs, metric_name),
            x_label=x_label,
            y_label=y_label,
            title=f"{stem.replace('_', ' ').title()}: {y_label}",
            output_path=output_root / "plots" / stem / plot_name,
        )
        plt.close(figure)


def _write_root_summary(
    path: Path,
    metrics: list[ExperimentMetrics],
    title: str,
) -> None:
    """Write the complete neutral experiment data summary."""

    text = "\n".join(
        (
            f"# {title}",
            "",
            "This report contains generated measurements only. Controller gains "
            "are fixed reference values recorded in each configuration file.",
            "",
            _markdown_table(metrics),
            "",
        )
    )
    path.write_text(text, encoding="utf-8")


def validate_experiment_outputs(
    output_root: Path,
    specs: tuple[ExperimentSpec, ...],
    metrics: list[ExperimentMetrics],
    include_hybrid: bool = False,
) -> None:
    """Validate the complete controller/scenario artifact set."""

    controller_names = (
        PHASE3_CONTROLLER_NAMES if include_hybrid else CONTROLLER_NAMES
    )
    expected_metric_count = len(specs) * len(controller_names)
    if len(metrics) != expected_metric_count:
        raise RuntimeError(
            "unexpected metric count: "
            f"expected {expected_metric_count}, got {len(metrics)}"
        )

    if set(controller_builders(include_hybrid)) != set(controller_names):
        raise RuntimeError("controller builder set does not match the official set")

    def require_file(path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty experiment artifact: {path}")

    for spec in specs:
        scenario_root = output_root / spec.group / spec.name
        required_files = (
            "configuration.json",
            "metrics.csv",
            "summary.md",
            "controller_comparison.png",
            *(
                f"{_controller_file_name(controller)}_trajectory.csv"
                for controller in controller_names
            ),
        )
        for filename in required_files:
            require_file(scenario_root / filename)

        scenario_metrics = [
            metric for metric in metrics if metric.experiment == spec.name
        ]
        if {metric.controller for metric in scenario_metrics} != set(
            controller_names
        ):
            raise RuntimeError(
                f"scenario does not contain all controllers: {spec.name}"
            )

    for filename in (
        "metrics.csv",
        "summary.md",
        "angle_sweep_metrics.csv",
        "disturbance_sweep_metrics.csv",
    ):
        require_file(output_root / filename)

    for filename in (
        "rmse.png",
        "wheel_drift.png",
        "control_effort.png",
    ):
        require_file(output_root / "plots" / "angle_sweep" / filename)
    for filename in (
        "deviation.png",
        "wheel_drift.png",
        "energy_deviation.png",
        "control_effort.png",
    ):
        require_file(output_root / "plots" / "disturbance_sweep" / filename)


def main(
    output_root: Path | None = None,
    summary_title: str = "PH-Balance Phase 2 experiment summary",
    include_hybrid: bool = False,
) -> Path:
    """Run every scenario and write reproducible output files."""

    output_root = (
        ROOT / "results" / "phase2" if output_root is None else Path(output_root)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    controller_parameters = nominal_parameters()
    specs = all_experiment_specs()
    all_metrics: list[ExperimentMetrics] = []
    for spec in specs:
        all_metrics.extend(
            run_spec(
                spec,
                output_root,
                controller_parameters,
                include_hybrid=include_hybrid,
            )
        )

    _write_metrics(output_root / "metrics.csv", all_metrics)
    _write_root_summary(output_root / "summary.md", all_metrics, summary_title)

    angle_specs = angle_sweep_specs()
    angle_metrics = _metrics_for_group(all_metrics, angle_specs)
    _write_sweep_outputs(
        output_root,
        angle_metrics,
        angle_specs,
        x_label="Initial lean angle (deg)",
        stem="angle_sweep",
        plots=(
            ("rmse_angle_deg", "Angle RMSE (deg)", "rmse.png"),
            ("wheel_drift_m", "Final wheel drift (m)", "wheel_drift.png"),
            (
                "integral_abs_torque_nm_s",
                "Integral absolute torque (N m s)",
                "control_effort.png",
            ),
        ),
    )

    disturbance_specs = disturbance_sweep_specs()
    disturbance_metrics = _metrics_for_group(
        all_metrics,
        disturbance_specs,
    )
    _write_sweep_outputs(
        output_root,
        disturbance_metrics,
        disturbance_specs,
        x_label="Disturbance amplitude (N m)",
        stem="disturbance_sweep",
        plots=(
            ("max_deviation_deg", "Maximum deviation (deg)", "deviation.png"),
            ("wheel_drift_m", "Final wheel drift (m)", "wheel_drift.png"),
            (
                "energy_deviation_rms_j",
                "Energy deviation RMS (J)",
                "energy_deviation.png",
            ),
            (
                "integral_abs_torque_nm_s",
                "Integral absolute torque (N m s)",
                "control_effort.png",
            ),
        ),
    )

    validate_experiment_outputs(
        output_root,
        specs,
        all_metrics,
        include_hybrid=include_hybrid,
    )

    print(
        f"Wrote {len(all_metrics)} controller measurements to {output_root}"
    )
    return output_root


if __name__ == "__main__":
    main()
