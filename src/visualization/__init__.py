"""Presentation-ready plots and animation boundaries."""

from .animation import create_robot_animation
from .plots import (
    plot_angle_response,
    plot_control_effort,
    plot_controller_comparison,
    plot_energy_evolution,
)

__all__ = [
    "create_robot_animation",
    "plot_angle_response",
    "plot_control_effort",
    "plot_controller_comparison",
    "plot_energy_evolution",
]
