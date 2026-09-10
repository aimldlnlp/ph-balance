"""Systematically search the implemented PH-V3 gain parameters.

The search is deliberately separate from the official comparison runner. It
uses a fixed Cartesian grid, rejects locally unstable or nonlinear candidates,
then ranks the surviving candidates across the three main Phase 2C scenarios.
No controller implementation is changed by this module.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import ExperimentMetrics, compute_metrics
from experiments.run_phase2_experiments import (
    TIMESTEP_S,
    ExperimentSpec,
    main_experiment_specs,
    nominal_parameters,
)
from src.controllers import PortHamiltonianV3Controller
from src.models.parameters import RobotParameters
from src.models.self_balancing_robot import state_derivative
from src.models.state import RobotState
from src.simulation import SimulationConfig, SimulationResult, Simulator


TORQUE_LIMIT = 3.0
MAX_SATURATION_FRACTION = 0.10
EIGENVALUE_MARGIN = 1.0e-6

ENERGY_GAINS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
ANGLE_GAINS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
POSITION_GAINS = (0.01, 0.03, 0.1, 0.3, 1.0)
DAMPING_GAINS = (0.001, 0.003, 0.01, 0.03, 0.1)


@dataclass(frozen=True)
class GainCandidate:
    """One PH-V3 gain tuple supported by the current controller interface."""

    energy_gain: float
    angle_gain: float
    position_gain: float
    damping_gain: float

    def build_controller(self) -> PortHamiltonianV3Controller:
        """Build a fresh unsaturated-state PH-V3 controller instance."""

        return PortHamiltonianV3Controller(
            energy_gain=self.energy_gain,
            angle_gain=self.angle_gain,
            position_gain=self.position_gain,
            damping_gain=self.damping_gain,
            desired_energy=0.0,
            torque_limit=TORQUE_LIMIT,
        )

    def as_dict(self) -> dict[str, float]:
        """Return CSV/JSON names matching the control notation."""

        return {
            "Ke": self.energy_gain,
            "Ktheta": self.angle_gain,
            "Ks": self.position_gain,
            "Kv": self.damping_gain,
        }


@dataclass(frozen=True)
class ScenarioMeasurement:
    """Metrics extended with wheel-position and saturation measurements."""

    metrics: ExperimentMetrics
    position_rms_m: float
    position_final_abs_m: float
    position_peak_abs_m: float
    saturation_fraction: float


def gain_grid() -> tuple[GainCandidate, ...]:
    """Return the complete deterministic Cartesian PH-V3 search grid."""

    return tuple(
        GainCandidate(energy, angle, position, damping)
        for energy in ENERGY_GAINS
        for angle in ANGLE_GAINS
        for position in POSITION_GAINS
        for damping in DAMPING_GAINS
    )


def _local_jacobian(
    candidate: GainCandidate,
    parameters: RobotParameters,
) -> np.ndarray:
    """Approximate the closed-loop state Jacobian at upright rest."""

    zero = np.zeros(4, dtype=float)
    step = 1.0e-6

    def right_hand_side(vector: np.ndarray) -> np.ndarray:
        state = RobotState(*map(float, vector))
        torque = candidate.build_controller().compute_torque(
            0.0,
            state,
            parameters,
        )
        return np.asarray(
            state_derivative(state, torque, parameters).as_tuple(),
            dtype=float,
        )

    return np.column_stack(
        [
            (
                right_hand_side(zero + step * np.eye(4)[index])
                - right_hand_side(zero - step * np.eye(4)[index])
            )
            / (2.0 * step)
            for index in range(4)
        ]
    )


def max_real_eigenvalue(
    candidate: GainCandidate,
    parameters: RobotParameters,
) -> float:
    """Return the largest real part of the local closed-loop eigenvalues."""

    return float(np.max(np.linalg.eigvals(_local_jacobian(candidate, parameters)).real))


def _saturation_fraction(result: SimulationResult) -> float:
    """Return the fraction of samples at the configured torque limit."""

    saturated = np.abs(result.control_input) >= TORQUE_LIMIT - 1.0e-9
    return float(np.mean(saturated))


def measure_candidate(
    candidate: GainCandidate,
    spec: ExperimentSpec,
    controller_parameters: RobotParameters,
) -> ScenarioMeasurement:
    """Simulate one candidate and calculate all tuning metrics."""

    result = Simulator(
        model=state_derivative,
        controller=candidate.build_controller(),
        parameters=spec.plant_parameters,
        controller_parameters=controller_parameters,
        config=SimulationConfig(
            duration=spec.duration,
            timestep=TIMESTEP_S,
            max_abs_angle=spec.max_abs_angle,
        ),
        disturbance=spec.disturbance,
    ).run(spec.initial_state)
    metrics = compute_metrics(
        result,
        experiment=spec.name,
        controller="PH-V3",
        disturbance_time=spec.disturbance_time,
    )
    position = result.state_history[:, 2]
    return ScenarioMeasurement(
        metrics=metrics,
        position_rms_m=float(np.sqrt(np.mean(position**2))),
        position_final_abs_m=float(abs(position[-1])),
        position_peak_abs_m=float(np.max(np.abs(position))),
        saturation_fraction=_saturation_fraction(result),
    )


def _nominal_score(measurement: ScenarioMeasurement) -> float:
    """Score nominal candidates with explicit normalized trade-off weights."""

    metrics = measurement.metrics
    settling_time = (
        metrics.settling_time_s
        if np.isfinite(metrics.settling_time_s)
        else 10.0
    )
    return float(
        metrics.rmse_angle_deg / 5.0
        + settling_time / 2.0
        + measurement.position_rms_m / 0.1
        + metrics.integral_abs_torque_nm_s / 1.0
    )


def _acceptable(measurement: ScenarioMeasurement) -> bool:
    """Return whether a candidate meets the nonlinear screening criteria."""

    return bool(
        measurement.metrics.stable
        and not measurement.metrics.termination_reason
        and measurement.saturation_fraction <= MAX_SATURATION_FRACTION
    )


def _reason(
    local_max_real: float,
    measurement: ScenarioMeasurement | None,
) -> str:
    """Explain why a candidate was rejected during nominal screening."""

    if local_max_real >= -EIGENVALUE_MARGIN:
        return "positive_or_zero_eigenvalue"
    if measurement is None:
        return "not_simulated"
    if measurement.metrics.termination_reason:
        return measurement.metrics.termination_reason
    if not measurement.metrics.stable:
        return "not_settled"
    if measurement.saturation_fraction > MAX_SATURATION_FRACTION:
        return "excessive_saturation"
    return ""


def _candidate_row(
    candidate: GainCandidate,
    local_max_real: float,
    measurement: ScenarioMeasurement | None,
    score: float = float("nan"),
) -> dict[str, float | int | str]:
    """Flatten one candidate into a reproducible search-result row."""

    gains = candidate.as_dict()
    if measurement is None:
        values: dict[str, float | int | str] = {
            "stable": 0,
            "termination_reason": "",
            "settling_time_s": float("nan"),
            "rmse_angle_deg": float("nan"),
            "position_rms_m": float("nan"),
            "position_final_abs_m": float("nan"),
            "position_peak_abs_m": float("nan"),
            "integral_abs_torque_nm_s": float("nan"),
            "saturation_fraction": float("nan"),
        }
    else:
        metrics = measurement.metrics
        values = {
            "stable": metrics.stable,
            "termination_reason": metrics.termination_reason,
            "settling_time_s": metrics.settling_time_s,
            "rmse_angle_deg": metrics.rmse_angle_deg,
            "position_rms_m": measurement.position_rms_m,
            "position_final_abs_m": measurement.position_final_abs_m,
            "position_peak_abs_m": measurement.position_peak_abs_m,
            "integral_abs_torque_nm_s": metrics.integral_abs_torque_nm_s,
            "saturation_fraction": measurement.saturation_fraction,
        }
    return {
        **gains,
        "max_real_eigenvalue": local_max_real,
        **values,
        "score": score,
        "rejected_reason": _reason(local_max_real, measurement),
    }


def _robustness_rows(
    candidate: GainCandidate,
    measurements: Iterable[ScenarioMeasurement],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    """Build scenario rows and an aggregate robustness ranking record."""

    rows: list[dict[str, float | int | str]] = []
    accepted_count = 0
    scores: list[float] = []
    rmse: list[float] = []
    position_rms: list[float] = []
    effort: list[float] = []
    saturation: list[float] = []
    for measurement in measurements:
        metrics = measurement.metrics
        accepted = _acceptable(measurement)
        accepted_count += int(accepted)
        score = _nominal_score(measurement)
        if not accepted:
            score += 20.0
        scores.append(score)
        rmse.append(metrics.rmse_angle_deg)
        position_rms.append(measurement.position_rms_m)
        effort.append(metrics.integral_abs_torque_nm_s)
        saturation.append(measurement.saturation_fraction)
        rows.append(
            {
                **candidate.as_dict(),
                "experiment": metrics.experiment,
                "stable": metrics.stable,
                "accepted": int(accepted),
                "termination_reason": metrics.termination_reason,
                "settling_time_s": metrics.settling_time_s,
                "rmse_angle_deg": metrics.rmse_angle_deg,
                "position_rms_m": measurement.position_rms_m,
                "position_final_abs_m": measurement.position_final_abs_m,
                "position_peak_abs_m": measurement.position_peak_abs_m,
                "integral_abs_torque_nm_s": metrics.integral_abs_torque_nm_s,
                "saturation_fraction": measurement.saturation_fraction,
                "scenario_score": score,
            }
        )
    summary = {
        **candidate.as_dict(),
        "accepted_scenarios": accepted_count,
        "robust_score": float(sum(scores)),
        "mean_rmse_angle_deg": float(np.mean(rmse)),
        "mean_position_rms_m": float(np.mean(position_rms)),
        "mean_integral_abs_torque_nm_s": float(np.mean(effort)),
        "max_saturation_fraction": float(np.max(saturation)),
    }
    return rows, summary


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write homogeneous tuning rows to CSV."""

    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: object) -> str:
    """Format scalar values for the Markdown tuning report."""

    if isinstance(value, float):
        return "-" if not np.isfinite(value) else f"{value:.6f}"
    return str(value)


def _write_summary(
    path: Path,
    grid_size: int,
    local_stable_count: int,
    nominal_feasible_count: int,
    best: dict[str, float | int | str],
    selected_measurements: list[ScenarioMeasurement],
) -> None:
    """Write search settings, selection logic, and best-candidate trade-offs."""

    lines = [
        "# PH-V3 systematic gain search",
        "",
        "This report was generated by a deterministic Cartesian search. "
        "No manual candidate was inserted into the ranking.",
        "",
        f"- Grid candidates: `{grid_size}`",
        f"- Local-eigenvalue candidates: `{local_stable_count}`",
        f"- Nominal nonlinear-feasible candidates: `{nominal_feasible_count}`",
        f"- Torque limit: `{TORQUE_LIMIT:.3f} N m`",
        f"- Excessive saturation threshold: `{MAX_SATURATION_FRACTION:.1%}`",
        "- Local filter: largest eigenvalue real part must be below "
        f"`{-EIGENVALUE_MARGIN:.1e} 1/s`",
        "- Nominal score: normalized angle RMSE + settling time + wheel "
        "position RMS + integrated absolute torque",
        "",
        "The task names an independent `Kd`, but the implemented PH-V3 "
        "interface exposes only `Ke`, `Ktheta`, `Ks`, and `Kv` "
        "(`damping_gain`). No fifth gain was invented for this search.",
        "",
        "## Best robust candidate",
        "",
        "| Ke | Ktheta | Ks | Kv | Accepted scenarios | Robust score | Mean angle RMSE (deg) | Mean position RMS (m) | Mean effort (N m s) | Max saturation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            _format(best[key])
            for key in (
                "Ke",
                "Ktheta",
                "Ks",
                "Kv",
                "accepted_scenarios",
                "robust_score",
                "mean_rmse_angle_deg",
                "mean_position_rms_m",
                "mean_integral_abs_torque_nm_s",
                "max_saturation_fraction",
            )
        )
        + " |",
        "",
        "## Best-candidate scenario results",
        "",
        "| Experiment | Stable | Settling (s) | Angle RMSE (deg) | Position RMS (m) | Final position error (m) | Effort (N m s) | Saturation | Termination |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for measurement in selected_measurements:
        metrics = measurement.metrics
        lines.append(
            "| "
            + " | ".join(
                (
                    metrics.experiment,
                    str(metrics.stable),
                    _format(metrics.settling_time_s),
                    _format(metrics.rmse_angle_deg),
                    _format(measurement.position_rms_m),
                    _format(measurement.position_final_abs_m),
                    _format(metrics.integral_abs_torque_nm_s),
                    _format(measurement.saturation_fraction),
                    metrics.termination_reason or "-",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "The ranking prioritizes the number of accepted main scenarios "
            "before the aggregate score. A failed or excessively saturated "
            "scenario receives a fixed score penalty, so a low-effort but "
            "unstable candidate cannot win.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> Path:
    """Run the PH-V3 search and write reproducible tuning evidence."""

    output = ROOT / "results" / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    nominal_parameters_set = nominal_parameters()
    main_specs = main_experiment_specs()
    nominal_spec = main_specs[0]

    search_rows: list[dict[str, float | int | str]] = []
    feasible: list[tuple[GainCandidate, ScenarioMeasurement, float]] = []
    local_stable_count = 0
    for candidate in gain_grid():
        local_max_real = max_real_eigenvalue(candidate, nominal_parameters_set)
        measurement: ScenarioMeasurement | None = None
        score = float("nan")
        if local_max_real < -EIGENVALUE_MARGIN:
            local_stable_count += 1
            measurement = measure_candidate(
                candidate,
                nominal_spec,
                nominal_parameters_set,
            )
            if _acceptable(measurement):
                score = _nominal_score(measurement)
                feasible.append((candidate, measurement, score))
        search_rows.append(
            _candidate_row(candidate, local_max_real, measurement, score)
        )

    if not feasible:
        raise RuntimeError("no PH-V3 candidate passed nominal screening")

    robustness_rows: list[dict[str, float | int | str]] = []
    robustness_summaries: list[dict[str, float | int | str]] = []
    measurements_by_candidate: dict[GainCandidate, list[ScenarioMeasurement]] = {}
    for candidate, _, _ in feasible:
        measurements = [
            measure_candidate(candidate, spec, nominal_parameters_set)
            for spec in main_specs
        ]
        measurements_by_candidate[candidate] = measurements
        rows, summary = _robustness_rows(candidate, measurements)
        robustness_rows.extend(rows)
        robustness_summaries.append(summary)

    best = sorted(
        robustness_summaries,
        key=lambda row: (-int(row["accepted_scenarios"]), float(row["robust_score"])),
    )[0]
    best_candidate = GainCandidate(
        energy_gain=float(best["Ke"]),
        angle_gain=float(best["Ktheta"]),
        position_gain=float(best["Ks"]),
        damping_gain=float(best["Kv"]),
    )
    selected_measurements = measurements_by_candidate[best_candidate]

    _write_rows(output / "phase2c_ph_v3_search.csv", search_rows)
    _write_rows(output / "phase2c_ph_v3_robustness.csv", robustness_rows)
    _write_rows(output / "phase2c_ph_v3_candidate_summary.csv", robustness_summaries)
    _write_rows(
        output / "phase2c_ph_v3_selected_validation.csv",
        [
            {
                **best_candidate.as_dict(),
                "experiment": measurement.metrics.experiment,
                "stable": measurement.metrics.stable,
                "settling_time_s": measurement.metrics.settling_time_s,
                "rmse_angle_deg": measurement.metrics.rmse_angle_deg,
                "position_rms_m": measurement.position_rms_m,
                "position_final_abs_m": measurement.position_final_abs_m,
                "position_peak_abs_m": measurement.position_peak_abs_m,
                "integral_abs_torque_nm_s": measurement.metrics.integral_abs_torque_nm_s,
                "saturation_fraction": measurement.saturation_fraction,
                "termination_reason": measurement.metrics.termination_reason,
            }
            for measurement in selected_measurements
        ],
    )
    (output / "phase2c_ph_v3_best.json").write_text(
        json.dumps(
            {
                "controller": "PH-V3",
                "gains": best_candidate.as_dict(),
                "torque_limit": TORQUE_LIMIT,
                "accepted_main_scenarios": best["accepted_scenarios"],
                "robust_score": best["robust_score"],
                "task_gain_note": "Independent Kd is not exposed by the current PH-V3 interface; Kv maps to damping_gain.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary(
        output / "phase2c_ph_v3_summary.md",
        len(search_rows),
        local_stable_count,
        len(feasible),
        best,
        selected_measurements,
    )
    print(
        f"Evaluated {len(search_rows)} candidates; wrote PH-V3 tuning evidence to {output}"
    )
    return output


if __name__ == "__main__":
    main()
