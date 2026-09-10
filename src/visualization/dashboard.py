"""Build a compact, portfolio-ready dashboard from Phase 3 CSV results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.visualization.plots import CONTROLLER_COLORS
from src.visualization.style import INK, TEAL, configure_style

CONTROLLERS = ("PID", "PH-V1", "PH-V2", "PH-V3", "Hybrid PH")
GROUPS = ("main", "angle_sweep", "disturbance_sweep")
GROUP_LABELS = {
    "main": "Main cases",
    "angle_sweep": "Angle sweep",
    "disturbance_sweep": "Disturbance sweep",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into rows while preserving the source columns."""

    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _number(value: str) -> float:
    """Convert a CSV scalar to float, including ``nan`` values."""

    return float(value)


def _controller_file_name(controller: str) -> str:
    """Return the trajectory filename stem used by the experiment runner."""

    return controller.lower().replace("-", "_").replace(" ", "_")


def _summary_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Index robustness rows by scenario group and controller."""

    return {(row["group"], row["controller"]): row for row in rows}


def _format_number(value: str, digits: int = 3) -> str:
    """Format a metric value or return a clear missing-value marker."""

    number = _number(value)
    return "-" if not np.isfinite(number) else f"{number:.{digits}f}"


def _draw_metric_bars(
    axis: Axes,
    summary: dict[tuple[str, str], dict[str, str]],
    field: str,
    title: str,
    y_label: str,
) -> None:
    """Draw a successful-run mean for the three main cases."""

    values = np.array(
        [
            _number(summary[("main", controller)][field])
            for controller in CONTROLLERS
        ],
        dtype=float,
    )
    finite = np.isfinite(values)
    positions = np.arange(len(CONTROLLERS))
    axis.bar(
        positions[finite],
        values[finite],
        color=[CONTROLLER_COLORS[controller] for controller in CONTROLLERS if finite[CONTROLLERS.index(controller)]],
        width=0.68,
    )

    finite_values = values[finite]
    upper_limit = max(float(np.max(finite_values)) * 1.25, 1.0) if finite_values.size else 1.0
    axis.set_ylim(0.0, upper_limit)
    for position, value in zip(positions, values):
        if np.isfinite(value):
            axis.text(
                position,
                value + upper_limit * 0.035,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
        else:
            axis.text(
                position,
                upper_limit * 0.05,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    axis.set_title(title, loc="left")
    axis.set_ylabel(y_label)
    axis.set_xticks(positions, CONTROLLERS, rotation=20, ha="right")
    axis.tick_params(length=0)


def _draw_success_rate(
    axis: Axes,
    summary: dict[tuple[str, str], dict[str, str]],
) -> None:
    """Draw success rate for each controller and scenario group."""

    positions = np.arange(len(GROUPS))
    width = 0.15
    for index, controller in enumerate(CONTROLLERS):
        values = np.array(
            [
                _number(summary[(group, controller)]["success_rate"])
                for group in GROUPS
            ]
        )
        bars = axis.bar(
            positions + (index - 2) * width,
            values,
            width=width,
            color=CONTROLLER_COLORS[controller],
            label=controller,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.025,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
    axis.set_ylim(0.0, 1.16)
    axis.set_title("Success rate by test group", loc="left")
    axis.set_ylabel("Completed runs")
    axis.set_xticks(positions, [GROUP_LABELS[group] for group in GROUPS])
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(ncol=2, fontsize=8, loc="upper center")
    axis.tick_params(length=0)


def _pulse_amplitude(experiment: str) -> float:
    """Extract the numeric disturbance amplitude from a scenario name."""

    return float(experiment.removeprefix("pulse_").removesuffix("_nm").replace("p", "."))


def _draw_disturbance_recovery(
    axis: Axes,
    metrics: list[dict[str, str]],
) -> None:
    """Draw recovery time and mark unsuccessful disturbance runs."""

    rows = [row for row in metrics if row["experiment"].startswith("pulse_")]
    amplitudes = sorted({_pulse_amplitude(row["experiment"]) for row in rows})
    all_recovery = [
        _number(row["recovery_time_s"])
        for row in rows
        if np.isfinite(_number(row["recovery_time_s"]))
    ]
    failure_level = max(all_recovery, default=1.0) * 1.15

    for controller in CONTROLLERS:
        controller_rows = {
            _pulse_amplitude(row["experiment"]): row
            for row in rows
            if row["controller"] == controller
        }
        good_x: list[float] = []
        good_y: list[float] = []
        failed_x: list[float] = []
        for amplitude in amplitudes:
            value = _number(controller_rows[amplitude]["recovery_time_s"])
            if np.isfinite(value):
                good_x.append(amplitude)
                good_y.append(value)
            else:
                failed_x.append(amplitude)

        axis.plot(
            good_x,
            good_y,
            marker="o",
            linewidth=2.0,
            color=CONTROLLER_COLORS[controller],
            label=controller,
        )
        if failed_x:
            axis.scatter(
                failed_x,
                [failure_level] * len(failed_x),
                marker="x",
                s=55,
                linewidths=2.0,
                color=CONTROLLER_COLORS[controller],
            )

    axis.axhline(
        failure_level,
        color=INK,
        linewidth=0.8,
        linestyle=":",
        alpha=0.65,
    )
    axis.scatter(
        [],
        [],
        marker="x",
        s=55,
        linewidths=2.0,
        color=INK,
        label="x = failed run",
    )
    axis.set_title("Disturbance recovery", loc="left")
    axis.set_xlabel("Pulse amplitude (N m)")
    axis.set_ylabel("Recovery time (s)")
    axis.set_xticks(amplitudes)
    axis.legend(ncol=2, fontsize=8, loc="upper left")
    axis.tick_params(length=0)


def _build_overview(
    metrics: list[dict[str, str]],
    robustness: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Create the six-panel dashboard overview."""

    configure_style()
    summary = _summary_index(robustness)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 8.5),
        constrained_layout=True,
    )
    figure.suptitle(
        "PH-Balance Phase 3 | Controller performance dashboard",
        x=0.04,
        ha="left",
        color=INK,
        fontsize=16,
    )

    _draw_success_rate(axes[0, 0], summary)
    _draw_metric_bars(
        axes[0, 1],
        summary,
        "mean_settling_successful_s",
        "Settling time | main cases",
        "Successful-run mean (s)",
    )
    _draw_metric_bars(
        axes[0, 2],
        summary,
        "mean_rmse_successful_deg",
        "Angle error | main cases",
        "Successful-run mean RMSE (deg)",
    )
    _draw_disturbance_recovery(axes[1, 0], metrics)
    _draw_metric_bars(
        axes[1, 1],
        summary,
        "mean_wheel_drift_successful_m",
        "Wheel drift | main cases",
        "Successful-run mean (m)",
    )
    _draw_metric_bars(
        axes[1, 2],
        summary,
        "mean_effort_successful_nm_s",
        "Control effort | main cases",
        "Successful-run mean (N m s)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _read_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Read time, angle, and wheel position from one trajectory CSV."""

    rows = _read_csv(path)
    return {
        "time": np.array([_number(row["time"]) for row in rows]),
        "theta": np.rad2deg(
            np.array([_number(row["theta"]) for row in rows])
        ),
        "wheel_position": np.array(
            [_number(row["wheel_position"]) for row in rows]
        ),
    }


def _build_trajectories(results_root: Path, output_path: Path) -> None:
    """Create a compact nominal/disturbance trajectory comparison."""

    configure_style()
    scenarios = (
        ("stabilization", "Nominal stabilization", False),
        ("disturbance_rejection", "1.0 N m disturbance", True),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 7.5),
        constrained_layout=True,
    )
    figure.suptitle(
        "PH-Balance closed-loop trajectories",
        x=0.04,
        ha="left",
        color=INK,
        fontsize=16,
    )

    handles = []
    labels = []
    for row_index, (scenario, title, has_disturbance) in enumerate(scenarios):
        scenario_root = results_root / "main" / scenario
        trajectories = {
            controller: _read_trajectory(
                scenario_root
                / f"{_controller_file_name(controller)}_trajectory.csv"
            )
            for controller in CONTROLLERS
        }
        for column_index, (field, y_label) in enumerate(
            (("theta", "Lean angle (deg)"), ("wheel_position", "Wheel position (m)"))
        ):
            axis = axes[row_index, column_index]
            for controller, trajectory in trajectories.items():
                line, = axis.plot(
                    trajectory["time"],
                    trajectory[field],
                    color=CONTROLLER_COLORS[controller],
                    linewidth=2.0,
                    label=controller,
                )
                if row_index == 0 and column_index == 0:
                    handles.append(line)
                    labels.append(controller)
            axis.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55)
            if has_disturbance:
                axis.axvline(
                    5.0,
                    color=TEAL,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.85,
                )
            axis.set_title(f"{title} | {y_label}", loc="left")
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(y_label)
            axis.tick_params(length=0)
            axis.margins(x=0.01)

    figure.legend(
        handles,
        labels,
        ncol=len(CONTROLLERS),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _success_table(
    metrics: list[dict[str, str]],
) -> list[str]:
    """Return Markdown rows for the grouped success-rate table."""

    lines = [
        "| Controller | Main | Angle sweep | Disturbance sweep | All 12 cases |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for controller in CONTROLLERS:
        cells: list[str] = []
        for experiments in (
            {"stabilization", "disturbance_rejection", "parameter_uncertainty"},
            {"theta_5_deg", "theta_15_deg", "theta_30_deg", "theta_45_deg"},
            {"pulse_0p2_nm", "pulse_0p5_nm", "pulse_1_nm", "pulse_1p5_nm", "pulse_2_nm"},
        ):
            group_rows = [
                row
                for row in metrics
                if row["controller"] == controller
                and row["experiment"] in experiments
            ]
            successes = sum(row["success"] == "1" for row in group_rows)
            cells.append(f"{successes}/{len(group_rows)}")
        successes = sum(
            row["controller"] == controller and row["success"] == "1"
            for row in metrics
        )
        lines.append(
            f"| {controller} | {cells[0]} | {cells[1]} | {cells[2]} | {successes}/12 |"
        )
    return lines


def _write_dashboard_markdown(
    results_root: Path,
    metrics: list[dict[str, str]],
    robustness: list[dict[str, str]],
) -> Path:
    """Write a concise narrative around the generated dashboard images."""

    summary = _summary_index(robustness)
    lines = [
        "# PH-Balance Phase 3 Results Dashboard",
        "",
        "A portfolio view of the five-controller comparison. The dashboard "
        "prioritizes success, angle regulation, disturbance recovery, wheel "
        "drift, and control effort.",
        "",
        "![Performance overview](dashboard/dashboard_overview.png)",
        "",
        "![Closed-loop trajectories](dashboard/trajectories.png)",
        "",
        "## How to read this dashboard",
        "",
        "- Success rate includes every configured run, including failed runs.",
        "- Continuous metric bars use successful-run means for the three main "
        "cases only.",
        "- In the recovery plot, `x` marks a run that did not produce a valid "
        "recovery time because it terminated or failed the success criteria.",
        "- Trajectories that stop early are visible; their shorter duration is "
        "not treated as lower effort or better performance.",
        "",
        "## Success-rate summary",
        "",
        *_success_table(metrics),
        "",
        "## Main-case performance",
        "",
        "Successful-run means; terminated runs are shown as `-`.",
        "",
        "| Controller | Angle RMSE (deg) | Settling (s) | Wheel drift (m) | Effort (N m s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for controller in CONTROLLERS:
        row = summary[("main", controller)]
        lines.append(
            "| "
            + " | ".join(
                (
                    controller,
                    _format_number(row["mean_rmse_successful_deg"]),
                    _format_number(row["mean_settling_successful_s"]),
                    _format_number(row["mean_wheel_drift_successful_m"]),
                    _format_number(row["mean_effort_successful_nm_s"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Practical reading",
            "",
            "- PID is the most reliable controller in this matrix: 12/12 "
            "successful runs, with faster angle settling and lower angle error.",
            "- Hybrid PH improves substantially over PH-V1, PH-V2, and PH-V3 "
            "and achieves lower final wheel drift in the main cases.",
            "- Hybrid PH completes all tested disturbance amplitudes, but its "
            "recovery time, angle error, and effort increase as pulse amplitude "
            "increases.",
            "- PH-V1 and PH-V2 fall over in every official Phase 3 scenario; "
            "their short trajectories must not be interpreted as low effort.",
            "",
            "## Source data",
            "",
            "- [All controller metrics](metrics.csv)",
            "- [Grouped robustness summary](robustness_summary.csv)",
            "- [Detailed Phase 3 analysis](final_report.md)",
            "",
            "Generated by `python -m experiments.build_phase3_dashboard`.",
        ]
    )
    output_path = results_root / "dashboard.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_phase3_dashboard(results_root: str | Path) -> Path:
    """Generate Phase 3 dashboard plots and return the Markdown path."""

    root = Path(results_root)
    metrics = _read_csv(root / "metrics.csv")
    robustness = _read_csv(root / "robustness_summary.csv")
    dashboard_root = root / "dashboard"
    _build_overview(
        metrics,
        robustness,
        dashboard_root / "dashboard_overview.png",
    )
    _build_trajectories(root, dashboard_root / "trajectories.png")
    return _write_dashboard_markdown(root, metrics, robustness)


__all__ = ["build_phase3_dashboard"]
