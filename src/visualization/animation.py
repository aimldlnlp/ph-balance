"""Sagittal robot animation for simulation trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon
import numpy as np

from src.models.parameters import RobotParameters
from src.simulation.simulator import SimulationResult
from src.visualization.style import BLUE, GRID, INK, TEAL, configure_style


def _body_vertices(
    wheel_position: float,
    theta: float,
    wheel_radius: float,
    body_length: float,
) -> np.ndarray:
    """Return a rectangular body aligned with the current lean angle."""

    direction = np.array([np.sin(theta), np.cos(theta)])
    normal = np.array([np.cos(theta), -np.sin(theta)])
    base = np.array([wheel_position, wheel_radius + 0.02]) + 0.04 * direction
    top = base + body_length * direction
    half_width = 0.07
    return np.array(
        [
            base + half_width * normal,
            base - half_width * normal,
            top - half_width * normal,
            top + half_width * normal,
        ]
    )


def create_robot_animation(
    result: SimulationResult,
    parameters: RobotParameters,
    output_path: str | Path,
    *,
    fps: int = 24,
    max_frames: int = 120,
) -> Path:
    """Render and save a clean sagittal robot animation as a GIF."""

    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive")

    configure_style()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    wheel_position = result.state_history[:, 2]
    theta = result.state_history[:, 0]
    frame_indices = np.unique(
        np.linspace(0, len(result.time) - 1, min(max_frames, len(result.time)), dtype=int)
    )
    wheel_radius = parameters.wheel_radius
    body_length = 2.0 * parameters.center_of_mass_length
    x_margin = max(0.45, 0.25 * (wheel_position.max() - wheel_position.min()))
    x_limits = (wheel_position.min() - x_margin, wheel_position.max() + x_margin)
    y_limits = (0.0, wheel_radius + body_length + 0.16)

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.set_title("PH-Balance - self-balancing robot", loc="left", color=INK, pad=12)
    axis.set_xlabel("Wheel position (m)")
    axis.set_ylabel("Height (m)")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(length=0)
    axis.axhline(0.0, color=GRID, linewidth=2.0)

    rear_wheel = Circle((wheel_position[0] - 0.055, wheel_radius), wheel_radius, color=GRID)
    front_wheel = Circle(
        (wheel_position[0] + 0.055, wheel_radius),
        wheel_radius,
        color=BLUE,
        alpha=0.9,
    )
    body = Polygon(
        _body_vertices(
            wheel_position[0],
            theta[0],
            wheel_radius,
            body_length,
        ),
        closed=True,
        facecolor=INK,
        edgecolor=INK,
        linewidth=1.0,
    )
    center_of_mass = axis.plot([], [], marker="o", color=TEAL, markersize=6)[0]
    axle = axis.plot([], [], color=TEAL, linewidth=3.0, solid_capstyle="round")[0]
    trail = axis.plot([], [], color=BLUE, linewidth=1.0, alpha=0.25)[0]
    time_text = axis.text(0.02, 0.94, "", transform=axis.transAxes, color=INK)
    angle_text = axis.text(0.02, 0.87, "", transform=axis.transAxes, color=INK)
    axis.add_patch(rear_wheel)
    axis.add_patch(front_wheel)
    axis.add_patch(body)

    def update(frame: int) -> tuple[object, ...]:
        """Update schematic geometry for one trajectory sample."""

        wheel = wheel_position[frame]
        lean = theta[frame]
        rear_wheel.center = (wheel - 0.055, wheel_radius)
        front_wheel.center = (wheel + 0.055, wheel_radius)
        body.set_xy(_body_vertices(wheel, lean, wheel_radius, body_length))
        center_x = wheel + parameters.center_of_mass_length * np.sin(lean)
        center_y = wheel_radius + parameters.center_of_mass_length * np.cos(lean)
        center_of_mass.set_data([center_x], [center_y])
        axle.set_data([wheel - 0.055, wheel + 0.055], [wheel_radius, wheel_radius])
        trail.set_data(wheel_position[: frame + 1], np.full(frame + 1, wheel_radius))
        time_text.set_text(f"t = {result.time[frame]:.2f} s")
        angle_text.set_text(f"theta = {np.rad2deg(lean):+.1f} deg")
        return (
            rear_wheel,
            front_wheel,
            body,
            center_of_mass,
            axle,
            trail,
            time_text,
            angle_text,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000.0 / fps,
        blit=True,
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(figure)
    return output


__all__ = ["create_robot_animation"]
