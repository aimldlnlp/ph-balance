"""Portfolio-quality plots for simulation results."""

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.simulation.simulator import SimulationResult
from src.visualization.style import (
    AMBER,
    BLUE,
    INK,
    TEAL,
    VIOLET,
    configure_style,
)

CONTROLLER_COLORS = {
    "PID": AMBER,
    "PH-V1": BLUE,
    "PH-V2": TEAL,
    "PH-V3": VIOLET,
    "Hybrid PH": INK,
    "Port-Hamiltonian": BLUE,
}


def _save_figure(figure: Figure, output_path: Optional[str | Path]) -> None:
    """Save a figure when a path is supplied."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _base_figure(title: str, y_label: str) -> tuple[Figure, plt.Axes]:
    """Create a consistently styled single-axis figure."""

    configure_style()
    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.set_title(title, loc="left", color=INK, pad=12)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(y_label)
    axis.tick_params(length=0)
    return figure, axis


def plot_angle_response(
    result: SimulationResult,
    output_path: Optional[str | Path] = None,
) -> Figure:
    """Plot body angle versus time in degrees."""

    figure, axis = _base_figure("Body angle response", "Lean angle (deg)")
    angle_degrees = np.rad2deg(result.state_history[:, 0])
    axis.plot(result.time, angle_degrees, color=BLUE, linewidth=2.2)
    axis.axhline(0.0, color=INK, linewidth=0.8, alpha=0.65)
    axis.margins(x=0.01)
    _save_figure(figure, output_path)
    return figure


def plot_energy_evolution(
    result: SimulationResult,
    output_path: Optional[str | Path] = None,
) -> Figure:
    """Plot the physical plant Hamiltonian versus time."""

    figure, axis = _base_figure(
        "Mechanical energy evolution",
        "Plant energy, H_phys (J)",
    )
    axis.plot(result.time, result.energy, color=TEAL, linewidth=2.2)
    axis.axhline(0.0, color=INK, linewidth=0.8, alpha=0.65)
    axis.margins(x=0.01)
    _save_figure(figure, output_path)
    return figure


def plot_control_effort(
    result: SimulationResult,
    output_path: Optional[str | Path] = None,
) -> Figure:
    """Plot commanded motor torque versus time."""

    figure, axis = _base_figure("Control effort", "Motor torque (N m)")
    axis.plot(result.time, result.control_input, color=AMBER, linewidth=2.0)
    axis.axhline(0.0, color=INK, linewidth=0.8, alpha=0.65)
    axis.margins(x=0.01)
    _save_figure(figure, output_path)
    return figure


def plot_controller_comparison(
    results: Mapping[str, SimulationResult],
    output_path: Optional[str | Path] = None,
    title: str = "Controller comparison",
    disturbance_time: Optional[float] = None,
) -> Figure:
    """Plot angle, energy, and commanded torque for several controllers."""

    if not results:
        raise ValueError("results must contain at least one controller")

    configure_style()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(title, x=0.08, ha="left", color=INK)
    for name, result in results.items():
        color = CONTROLLER_COLORS.get(name, TEAL)
        label = name.replace("_", " ")
        axes[0].plot(
            result.time,
            np.rad2deg(result.state_history[:, 0]),
            color=color,
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            result.time,
            result.energy,
            color=color,
            linewidth=2.0,
            label=label,
        )
        axes[2].plot(
            result.time,
            result.control_input,
            color=color,
            linewidth=2.0,
            label=label,
        )

    axes[0].set_ylabel("Theta (deg)")
    axes[1].set_ylabel("Plant energy, H_phys (J)")
    axes[2].set_ylabel("Torque (N m)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.axhline(0.0, color=INK, linewidth=0.8, alpha=0.55)
        if disturbance_time is not None:
            axis.axvline(
                disturbance_time,
                color=TEAL,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
        axis.tick_params(length=0)
        axis.legend(frameon=False, loc="best")
        axis.margins(x=0.01)

    _save_figure(figure, output_path)
    return figure


def plot_sweep_metric(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Optional[str | Path] = None,
) -> Figure:
    """Plot one scalar metric against a sweep variable for each controller."""

    if not series:
        raise ValueError("series must contain at least one controller")

    figure, axis = _base_figure(title, y_label)
    for name, (x_values, y_values) in series.items():
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            label=name.replace("_", " "),
            color=CONTROLLER_COLORS.get(name, TEAL),
        )
    axis.set_xlabel(x_label)
    axis.legend(frameon=False, loc="best")
    axis.margins(x=0.03)
    _save_figure(figure, output_path)
    return figure


__all__ = [
    "plot_angle_response",
    "plot_control_effort",
    "plot_controller_comparison",
    "plot_energy_evolution",
    "plot_sweep_metric",
]
