"""Generate the official Phase 3 Hybrid PH comparison artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.run_phase2_experiments import (
    PHASE3_CONTROLLER_NAMES,
    ROOT,
    main as run_experiments,
)


ROBUSTNESS_GROUPS = {
    "main": {"stabilization", "disturbance_rejection", "parameter_uncertainty"},
    "angle_sweep": {
        "theta_5_deg",
        "theta_15_deg",
        "theta_30_deg",
        "theta_45_deg",
    },
    "disturbance_sweep": {
        "pulse_0p2_nm",
        "pulse_0p5_nm",
        "pulse_1_nm",
        "pulse_1p5_nm",
        "pulse_2_nm",
    },
}


def _mean_successful(rows: list[dict[str, str]], field: str) -> float:
    """Return a metric mean using completed runs only."""

    values = [float(row[field]) for row in rows if row["success"] == "1"]
    return float(np.mean(values)) if values else float("nan")


def _write_robustness_results(output_root: Path) -> None:
    """Write fair success-rate and successful-run robustness summaries."""

    with (output_root / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        rows = list(csv.DictReader(source))

    summary_rows: list[dict[str, str | float | int]] = []
    for group, experiment_names in ROBUSTNESS_GROUPS.items():
        group_rows = [
            row for row in rows if row["experiment"] in experiment_names
        ]
        for controller in PHASE3_CONTROLLER_NAMES:
            controller_rows = [
                row for row in group_rows if row["controller"] == controller
            ]
            successes = [
                row for row in controller_rows if row["success"] == "1"
            ]
            summary_rows.append(
                {
                    "group": group,
                    "controller": controller,
                    "scenario_count": len(controller_rows),
                    "success_count": len(successes),
                    "success_rate": (
                        len(successes) / len(controller_rows)
                        if controller_rows
                        else float("nan")
                    ),
                    "fallover_count": sum(
                        row["termination_reason"] == "angle_limit"
                        for row in controller_rows
                    ),
                    "mean_rmse_successful_deg": _mean_successful(
                        controller_rows,
                        "rmse_angle_deg",
                    ),
                    "mean_settling_successful_s": _mean_successful(
                        controller_rows,
                        "settling_time_s",
                    ),
                    "mean_wheel_drift_successful_m": _mean_successful(
                        controller_rows,
                        "wheel_drift_m",
                    ),
                    "mean_effort_successful_nm_s": _mean_successful(
                        controller_rows,
                        "integral_abs_torque_nm_s",
                    ),
                }
            )

    output_path = output_root / "robustness_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Phase 3 robustness summary",
        "",
        "Successful-run means exclude terminated runs; fallover counts and "
        "success rates are reported separately.",
        "",
        "| " + " | ".join(summary_rows[0]) + " |",
        "| " + " | ".join("---" for _ in summary_rows[0]) + " |",
    ]
    for row in summary_rows:
        values = []
        for value in row.values():
            if isinstance(value, float):
                values.append(
                    "-" if not np.isfinite(value) else f"{value:.6f}"
                )
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    (output_root / "robustness_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> Path:
    """Run the five-controller Phase 3 matrix and robustness summaries."""

    output_root = run_experiments(
        output_root=ROOT / "results" / "phase3",
        summary_title="PH-Balance Phase 3 Hybrid PH experiment summary",
        include_hybrid=True,
    )
    _write_robustness_results(output_root)
    print(f"Wrote robustness summaries to {output_root}")
    return output_root


if __name__ == "__main__":
    main()
