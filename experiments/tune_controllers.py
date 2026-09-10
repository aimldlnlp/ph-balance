"""Perform small, staged controller-gain studies.

The PID study changes one gain family at a time. The Port-Hamiltonian study
checks a bounded grid because its two gains are coupled through the energy and
power-port terms. Results are written under ``results/tuning``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import ExperimentMetrics, compute_metrics
from experiments.run_experiments import (
    PH_REFERENCE_GAINS,
    PID_GAINS,
    ExperimentSpec,
    experiment_specs,
    nominal_parameters,
)
from src.controllers.pid import PIDController
from src.controllers.port_hamiltonian import PortHamiltonianController
from src.controllers.protocols import Controller
from src.models.parameters import RobotParameters
from src.models.self_balancing_robot import state_derivative
from src.simulation import SimulationConfig, SimulationResult, Simulator


def _simulate(
    controller: Controller,
    spec: ExperimentSpec,
    controller_parameters: RobotParameters,
) -> ExperimentMetrics:
    """Run one candidate on the nominal stabilization scenario."""

    result: SimulationResult = Simulator(
        model=state_derivative,
        controller=controller,
        parameters=spec.plant_parameters,
        controller_parameters=controller_parameters,
        config=SimulationConfig(
            spec.duration,
            spec.timestep,
            max_abs_angle=spec.max_abs_angle,
        ),
        disturbance=spec.disturbance,
    ).run(spec.initial_state)
    return compute_metrics(
        result,
        experiment=spec.name,
        controller=type(controller).__name__,
        disturbance_time=spec.disturbance_time,
    )


def _row(
    stage: str,
    gains: dict[str, float],
    metrics: ExperimentMetrics,
) -> dict[str, str | float | int]:
    """Merge candidate gains and measured metrics into one CSV row."""

    return {"stage": stage, **gains, **metrics.as_dict()}


def _pid_rows(spec, parameters) -> list[dict[str, str | float | int]]:
    """Tune PID in proportional, derivative, then integral stages."""

    rows: list[dict[str, str | float | int]] = []
    for kp in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0):
        gains = {"Kp": kp, "Ki": 0.0, "Kd": 2.0}
        rows.append(
            _row(
                "proportional_sweep",
                gains,
                _simulate(
                    PIDController(
                        angle_kp=kp,
                        angle_ki=0.0,
                        angle_kd=2.0,
                        torque_limit=3.0,
                        angle_reference_limit=0.3,
                    ),
                    spec,
                    parameters,
                ),
            )
        )

    for kd in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        gains = {"Kp": 30.0, "Ki": 0.0, "Kd": kd}
        rows.append(
            _row(
                "derivative_sweep",
                gains,
                _simulate(
                    PIDController(
                        angle_kp=30.0,
                        angle_ki=0.0,
                        angle_kd=kd,
                        torque_limit=3.0,
                        angle_reference_limit=0.3,
                    ),
                    spec,
                    parameters,
                ),
            )
        )

    for ki in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        gains = {"Kp": 30.0, "Ki": ki, "Kd": 3.0}
        rows.append(
            _row(
                "integral_sweep",
                gains,
                _simulate(
                    PIDController(
                        angle_kp=30.0,
                        angle_ki=ki,
                        angle_kd=3.0,
                        torque_limit=3.0,
                        angle_reference_limit=0.3,
                    ),
                    spec,
                    parameters,
                ),
            )
        )
    return rows


def _ph_rows(spec, parameters) -> list[dict[str, str | float | int]]:
    """Evaluate physically bounded positive-gain Port-Hamiltonian candidates."""

    rows: list[dict[str, str | float | int]] = []
    for energy_gain in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        for damping_gain in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            gains = {"Ke": energy_gain, "Kd": damping_gain}
            rows.append(
                _row(
                    "positive_gain_grid",
                    gains,
                    _simulate(
                        PortHamiltonianController(
                            energy_gain=energy_gain,
                            damping_gain=damping_gain,
                            torque_limit=3.0,
                        ),
                        spec,
                        parameters,
                    ),
                )
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    """Write tuning rows using the first row's stable field order."""

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_selected_gains(
    parameters: RobotParameters,
) -> list[ExperimentMetrics]:
    """Validate the selected PID and reference PH settings across scenarios."""

    metrics: list[ExperimentMetrics] = []
    for spec in experiment_specs():
        for name, controller in (
            ("PID", PIDController(**PID_GAINS)),
            (
                "Port-Hamiltonian",
                PortHamiltonianController(**PH_REFERENCE_GAINS),
            ),
        ):
            result: SimulationResult = Simulator(
                model=state_derivative,
                controller=controller,
                parameters=spec.plant_parameters,
                controller_parameters=parameters,
                config=SimulationConfig(
                    spec.duration,
                    spec.timestep,
                    max_abs_angle=spec.max_abs_angle,
                ),
                disturbance=spec.disturbance,
            ).run(spec.initial_state)
            metrics.append(
                compute_metrics(
                    result,
                    experiment=spec.name,
                    controller=name,
                    disturbance_time=spec.disturbance_time,
                )
            )
    return metrics


def _write_summary(
    path: Path,
    ph_rows: list[dict[str, str | float | int]],
    validation: list[ExperimentMetrics],
) -> None:
    """Write the tuning rationale and selected-gain validation table."""

    ph_stable = [row for row in ph_rows if row["stable"] == 1]
    lines = [
        "# Controller tuning",
        "",
        "PID was tuned in three staged sweeps: proportional authority, "
        "derivative damping, then a small integral term. The selected gains "
        "were chosen as a speed/effort trade-off rather than the most "
        "aggressive response.",
        "",
        "Selected PID: `Kp=30.0`, `Ki=0.5`, `Kd=3.0`.",
        "",
        "The Port-Hamiltonian grid used non-negative gains and the same "
        "nominal +15 degree rest condition. "
        f"Feasible stabilizing pairs found: `{len(ph_stable)}`.",
        "",
        "At the initial condition, `H>0` and the actuator-port velocity is "
        "zero, so the implemented `u=-Ke(H-Hd)-Kd*y` law commands negative "
        "torque for every `Ke>0`. The plant requires positive corrective "
        "torque for this lean direction. No positive `Ke/Kd` tuning can "
        "therefore meet the stabilization objective without changing the "
        "controller sign/structure.",
        "",
        "The current `Ke=0.1`, `Kd=1.0` pair is retained only as a bounded "
        "reference for the comparison plots; it is not reported as a valid "
        "stabilizing tune.",
        "",
        "Rows terminated by `angle_limit` have shorter horizons; their effort "
        "values are diagnostic and must not be interpreted as controller "
        "efficiency.",
        "",
        "## Selected-gain validation",
        "",
        "| Experiment | Controller | Duration (s) | Settling (s) | RMSE (deg) | Recovery (s) | Mean abs torque (N m) | Stable | Termination |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in validation:
        values = metric.as_dict()
        def fmt(value: object) -> str:
            if isinstance(value, float):
                return "-" if not np.isfinite(value) else f"{value:.3f}"
            if isinstance(value, str) and not value:
                return "-"
            return str(value)

        lines.append(
            f"| {values['experiment']} | {values['controller']} | "
            f"{fmt(values['simulated_duration_s'])} | "
            f"{fmt(values['settling_time_s'])} | "
            f"{fmt(values['rmse_angle_deg'])} | "
            f"{fmt(values['recovery_time_s'])} | "
            f"{fmt(values['mean_abs_torque_nm'])} | "
            f"{values['stable']} | {fmt(values['termination_reason'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run staged tuning and write reproducible evidence."""

    output = ROOT / "results" / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    parameters = nominal_parameters()
    nominal = experiment_specs()[0]
    pid_rows = _pid_rows(nominal, parameters)
    ph_rows = _ph_rows(nominal, parameters)
    validation = _validate_selected_gains(parameters)

    _write_csv(output / "pid_tuning.csv", pid_rows)
    _write_csv(output / "port_hamiltonian_tuning.csv", ph_rows)
    _write_csv(
        output / "selected_validation.csv",
        [metric.as_dict() for metric in validation],
    )
    (output / "selected_gains.json").write_text(
        json.dumps(
            {
                "PID": {
                    "Kp": PID_GAINS["angle_kp"],
                    "Ki": PID_GAINS["angle_ki"],
                    "Kd": PID_GAINS["angle_kd"],
                    "status": "selected",
                },
                "Port-Hamiltonian": {
                    "Ke": PH_REFERENCE_GAINS["energy_gain"],
                    "Kd": PH_REFERENCE_GAINS["damping_gain"],
                    "status": "reference_only_no_feasible_positive_gain_pair",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary(output / "summary.md", ph_rows, validation)
    print(f"Wrote tuning evidence to {output}")


if __name__ == "__main__":
    main()
