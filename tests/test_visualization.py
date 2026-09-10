"""Unit tests for portfolio visualization output."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.models import RobotParameters
from src.simulation import SimulationResult
from src.visualization import (
    create_robot_animation,
    plot_angle_response,
    plot_control_effort,
    plot_energy_evolution,
)


def make_parameters() -> RobotParameters:
    """Return deterministic robot geometry for animation tests."""

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


def make_result() -> SimulationResult:
    """Return a short valid trajectory for rendering tests."""

    return SimulationResult(
        time=np.array([0.0, 0.1, 0.2]),
        state_history=np.array(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.05, -0.2, 0.01, 0.1],
                [0.0, -0.1, 0.02, 0.05],
            ]
        ),
        control_input=np.array([0.5, 0.2, 0.2]),
        energy=np.array([0.05, 0.02, 0.01]),
    )


class VisualizationTests(unittest.TestCase):
    """Check that figures and GIFs are generated without visualization coupling."""

    def test_required_plots_are_written(self) -> None:
        """Angle, energy, and control figures save to requested paths."""

        result = make_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figures = [
                plot_angle_response(result, root / "angle.png"),
                plot_energy_evolution(result, root / "energy.png"),
                plot_control_effort(result, root / "torque.png"),
            ]
            for figure in figures:
                plt.close(figure)

            self.assertTrue((root / "angle.png").is_file())
            self.assertTrue((root / "energy.png").is_file())
            self.assertTrue((root / "torque.png").is_file())

    def test_robot_animation_is_written(self) -> None:
        """The sagittal animation saves as a GIF."""

        with tempfile.TemporaryDirectory() as directory:
            path = create_robot_animation(
                make_result(),
                make_parameters(),
                Path(directory) / "balance.gif",
                fps=5,
                max_frames=3,
            )

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
