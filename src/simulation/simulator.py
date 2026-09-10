"""SciPy-based simulation framework for PH-Balance.

The simulator integrates a model derivative with
``scipy.integrate.solve_ivp``. The controller is sampled once per output
interval and its command is held constant during that interval. This
zero-order hold prevents adaptive solver stages from advancing stateful
controller memory multiple times.

The model callable uses the project's four-state ordering. Four-argument
models accept ``external_torque`` as a generalized plant disturbance;
legacy three-argument models remain valid when no disturbance is requested.
"""

import csv
from dataclasses import dataclass
from inspect import signature
from math import isfinite
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.controllers.protocols import Controller
from src.models.hamiltonian import physical_total_energy
from src.models.parameters import RobotParameters
from src.models.state import RobotState

FloatArray = NDArray[np.float64]
RobotModel = (
    Callable[[RobotState, float, RobotParameters], RobotState]
    | Callable[[RobotState, float, RobotParameters, float], RobotState]
)
Disturbance = Callable[[float], float]


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and output settings for one simulation run."""

    duration: float
    timestep: float
    method: str = "RK45"
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-10
    max_abs_angle: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid integration settings at the boundary."""

        for name, value in (
            ("duration", self.duration),
            ("timestep", self.timestep),
            ("relative_tolerance", self.relative_tolerance),
            ("absolute_tolerance", self.absolute_tolerance),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.method:
            raise ValueError("method must not be empty")
        if self.max_abs_angle is not None and (
            not isfinite(self.max_abs_angle) or self.max_abs_angle <= 0.0
        ):
            raise ValueError("max_abs_angle must be positive and finite")


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory, input, and physical energy history from one run.

    ``state_history`` columns are ``[theta, theta_dot, wheel_position,
    wheel_velocity]``. ``control_input[i]`` is the command held over the
    interval beginning at ``time[i]``; ``disturbance[i]`` follows the same
    convention. The final entries repeat the previous values so all histories
    have equal length. ``termination_reason`` is set when a configured
    simulation validity event stops integration early.
    """

    time: FloatArray
    state_history: FloatArray
    control_input: FloatArray
    energy: FloatArray
    disturbance: FloatArray | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate trajectory dimensions and finite recorded values."""

        time = np.asarray(self.time, dtype=float)
        state_history = np.asarray(self.state_history, dtype=float)
        control_input = np.asarray(self.control_input, dtype=float)
        energy = np.asarray(self.energy, dtype=float)
        disturbance = (
            np.zeros_like(time)
            if self.disturbance is None
            else np.asarray(self.disturbance, dtype=float)
        )

        if time.ndim != 1:
            raise ValueError("time must be one-dimensional")
        if state_history.ndim != 2 or state_history.shape[1] != 4:
            raise ValueError("state_history must have shape (N, 4)")
        if len(time) != len(state_history):
            raise ValueError("time and state_history must have equal length")
        if (
            len(control_input) != len(time)
            or len(energy) != len(time)
            or len(disturbance) != len(time)
        ):
            raise ValueError("all histories must have equal length")
        if len(time) > 1 and not np.all(np.diff(time) > 0.0):
            raise ValueError("time must be strictly increasing")
        if not (
            np.isfinite(time).all()
            and np.isfinite(state_history).all()
            and np.isfinite(control_input).all()
            and np.isfinite(energy).all()
            and np.isfinite(disturbance).all()
        ):
            raise ValueError("simulation results must contain finite values")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "state_history", state_history)
        object.__setattr__(self, "control_input", control_input)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "disturbance", disturbance)

    def export_csv(self, path: str | Path) -> None:
        """Export time, state, input, disturbance, and physical energy."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "time",
                    "theta",
                    "theta_dot",
                    "wheel_position",
                    "wheel_velocity",
                    "control_input",
                    "disturbance_torque",
                    "energy",
                ]
            )
            for index, timestamp in enumerate(self.time):
                writer.writerow(
                    [
                        timestamp,
                        *self.state_history[index],
                        self.control_input[index],
                        self.disturbance[index],
                        self.energy[index],
                    ]
                )


def _state_from_vector(vector: np.ndarray) -> RobotState:
    """Convert a solver vector in project ordering to ``RobotState``."""

    if vector.shape != (4,):
        raise ValueError("solver state must have dimension 4")
    if not np.isfinite(vector).all():
        raise ValueError("solver state must be finite")
    return RobotState(
        theta=float(vector[0]),
        theta_dot=float(vector[1]),
        wheel_position=float(vector[2]),
        wheel_velocity=float(vector[3]),
    )


def _state_vector(state: RobotState) -> FloatArray:
    """Convert ``RobotState`` to a solver-compatible vector."""

    vector = np.asarray(state.as_tuple(), dtype=float)
    if vector.shape != (4,) or not np.isfinite(vector).all():
        raise ValueError("state must contain four finite values")
    return vector


def _accepts_disturbance(model: RobotModel) -> bool:
    """Check whether a model exposes the optional fourth argument."""

    try:
        return len(signature(model).parameters) >= 4
    except (TypeError, ValueError):
        # Callable objects without inspectable signatures use the current
        # four-argument model contract.
        return True


class Simulator:
    """Integrate one robot/controller pair and record its trajectory."""

    def __init__(
        self,
        model: RobotModel,
        controller: Controller,
        parameters: RobotParameters,
        config: SimulationConfig,
        disturbance: Disturbance | None = None,
        controller_parameters: RobotParameters | None = None,
    ) -> None:
        """Create a simulator without starting integration.

        ``parameters`` describe the plant. ``controller_parameters`` can stay
        nominal while the plant parameters are perturbed for robustness tests.
        """

        self.model = model
        self.controller = controller
        self.parameters = parameters
        self.config = config
        self.disturbance = disturbance
        self._model_accepts_disturbance = _accepts_disturbance(model)
        self.controller_parameters = (
            parameters if controller_parameters is None else controller_parameters
        )

    def run(self, initial_state: RobotState) -> SimulationResult:
        """Integrate from ``initial_state`` and return recorded histories."""

        initial_vector = _state_vector(initial_state)
        if (
            self.config.max_abs_angle is not None
            and abs(initial_state.theta) >= self.config.max_abs_angle
        ):
            raise ValueError("initial angle exceeds max_abs_angle")
        number_of_steps = int(np.ceil(self.config.duration / self.config.timestep))
        scheduled_time = np.linspace(
            0.0,
            self.config.duration,
            number_of_steps + 1,
        )
        recorded_time: list[float] = [0.0]
        state_vectors: list[FloatArray] = [initial_vector.copy()]
        controls: list[float] = []
        disturbances: list[float] = []
        termination_reason: str | None = None

        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()

        for index in range(number_of_steps):
            start_time = float(scheduled_time[index])
            end_time = float(scheduled_time[index + 1])
            step = end_time - start_time
            current_state = _state_from_vector(state_vectors[-1])
            control = float(
                self.controller.compute_torque(
                    start_time,
                    current_state,
                    self.controller_parameters,
                )
            )
            if not np.isfinite(control):
                raise ValueError("controller returned a non-finite torque")
            controls.append(control)
            external_torque = 0.0 if self.disturbance is None else float(
                self.disturbance(start_time)
            )
            if not np.isfinite(external_torque):
                raise ValueError("disturbance returned a non-finite torque")
            disturbances.append(external_torque)

            def dynamics(_time: float, vector: np.ndarray) -> FloatArray:
                """Evaluate the plant with the held command."""

                current_state = _state_from_vector(vector)
                if self._model_accepts_disturbance:
                    derivative = self.model(
                        current_state,
                        control,
                        self.parameters,
                        external_torque,
                    )
                else:
                    if external_torque != 0.0:
                        raise TypeError(
                            "a disturbed simulation requires a four-argument model"
                        )
                    derivative = self.model(
                        current_state,
                        control,
                        self.parameters,
                    )
                return _state_vector(derivative)

            events = None
            if self.config.max_abs_angle is not None:
                max_abs_angle = self.config.max_abs_angle

                def angle_limit_event(
                    _time: float,
                    vector: np.ndarray,
                ) -> float:
                    """Stop when the local upright model reaches fallover."""

                    return max_abs_angle - abs(float(vector[0]))

                angle_limit_event.terminal = True
                angle_limit_event.direction = -1
                events = angle_limit_event

            solution = solve_ivp(
                dynamics,
                (start_time, end_time),
                state_vectors[-1],
                method=self.config.method,
                rtol=self.config.relative_tolerance,
                atol=self.config.absolute_tolerance,
                max_step=step,
                events=events,
            )
            if not solution.success:
                raise RuntimeError(f"integration failed: {solution.message}")
            state_vectors.append(np.asarray(solution.y[:, -1], dtype=float))
            recorded_time.append(float(solution.t[-1]))
            if events is not None and solution.t_events[0].size:
                termination_reason = "angle_limit"
                break

        state_history = np.vstack(state_vectors)
        time = np.asarray(recorded_time, dtype=float)
        control_input = np.asarray(controls + [controls[-1]], dtype=float)
        disturbance_history = np.asarray(
            disturbances + [disturbances[-1]],
            dtype=float,
        )
        energy = np.asarray(
            [
                physical_total_energy(_state_from_vector(vector), self.parameters)
                for vector in state_history
            ],
            dtype=float,
        )
        return SimulationResult(
            time,
            state_history,
            control_input,
            energy,
            disturbance_history,
            termination_reason,
        )


__all__ = [
    "RobotModel",
    "Disturbance",
    "SimulationConfig",
    "SimulationResult",
    "Simulator",
]
