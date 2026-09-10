"""Generate the short visual assets used by the portfolio showcase."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.run_phase2_experiments import nominal_parameters
from src.simulation import SimulationResult
from src.visualization.animation import create_robot_animation

ROOT = Path(__file__).resolve().parents[1]


def _load_trajectory(path: Path) -> SimulationResult:
    """Load one exported trajectory into the animation result contract."""

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"trajectory is empty: {path}")

    def column(name: str) -> np.ndarray:
        """Return one finite numeric CSV column."""

        values = np.array([float(row[name]) for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"trajectory column is not finite: {name}")
        return values

    state_history = np.column_stack(
        [
            column("theta"),
            column("theta_dot"),
            column("wheel_position"),
            column("wheel_velocity"),
        ]
    )
    return SimulationResult(
        time=column("time"),
        state_history=state_history,
        control_input=column("control_input"),
        energy=column("energy"),
        disturbance=column("disturbance_torque"),
    )


def main() -> tuple[Path, Path]:
    """Render nominal balancing and 1.0 N m disturbance-recovery GIFs."""

    phase3 = ROOT / "results" / "phase3" / "main"
    animation_root = ROOT / "results" / "animations"
    showcase_root = ROOT / "results" / "showcase"
    nominal = _load_trajectory(
        phase3 / "stabilization" / "hybrid_ph_trajectory.csv"
    )
    disturbance = _load_trajectory(
        phase3 / "disturbance_rejection" / "hybrid_ph_trajectory.csv"
    )
    parameters = nominal_parameters()
    nominal_path = create_robot_animation(
        nominal,
        parameters,
        animation_root / "balance.gif",
        fps=20,
        max_frames=120,
    )
    disturbance_path = create_robot_animation(
        disturbance,
        parameters,
        showcase_root / "disturbance_recovery.gif",
        fps=20,
        max_frames=160,
    )
    print(f"Wrote nominal animation to {nominal_path}")
    print(f"Wrote disturbance animation to {disturbance_path}")
    return nominal_path, disturbance_path


if __name__ == "__main__":
    main()
