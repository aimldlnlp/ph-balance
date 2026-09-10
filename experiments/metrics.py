"""Metrics used by the reproducible controller experiments."""

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

import numpy as np

from src.simulation.simulator import SimulationResult


@dataclass(frozen=True)
class ExperimentMetrics:
    """Scalar performance summary for one controller/scenario pair.

    Torque effort is evaluated using the simulator's zero-order hold:
    ``sum(abs(u_i) * dt_i)``. Energy fields describe time-weighted deviation
    from a reference sample, not convergence to a target. ``wheel_drift_m``
    is the absolute final wheel position relative to the zero-position target.
    ``success`` is the primary completion flag; ``stable`` remains a
    compatibility alias.
    """

    experiment: str
    controller: str
    simulated_duration_s: float
    settling_time_s: float
    rmse_angle_deg: float
    peak_angle_deg: float
    overshoot_deg: float
    max_deviation_deg: float
    recovery_time_s: float
    peak_torque_nm: float
    rms_torque_nm: float
    mean_abs_torque_nm: float
    integral_abs_torque_nm_s: float
    final_angle_deg: float
    wheel_drift_m: float
    final_energy_j: float
    success: int
    termination_reason: str = ""
    termination_time_s: float = float("nan")
    fallover_penalty: float = 0.0
    energy_reference_j: float = float("nan")
    energy_deviation_rms_j: float = float("nan")
    final_energy_deviation_j: float = float("nan")

    @property
    def stable(self) -> int:
        """Return the legacy name for the success flag."""

        return self.success

    def as_dict(self) -> dict[str, str | float | int]:
        """Return CSV-friendly column names and values."""

        values = asdict(self)
        # Keep existing consumers readable while making ``success`` primary.
        values["stable"] = self.success
        return values


def _settling_time(
    time: np.ndarray,
    angle_degrees: np.ndarray,
    tolerance_degrees: float,
    start_time: float = 0.0,
) -> float:
    """Find the first time after which the angle stays in the tolerance band."""

    candidates = np.flatnonzero(time >= start_time)
    for index in candidates:
        if np.all(np.abs(angle_degrees[index:]) <= tolerance_degrees):
            return float(time[index])
    return float("nan")


def compute_metrics(
    result: SimulationResult,
    experiment: str,
    controller: str,
    disturbance_time: float | None = None,
    tolerance_degrees: float = 2.0,
) -> ExperimentMetrics:
    """Compute stabilization, rejection, and effort metrics.

    Settling is defined as the first sampled time after which ``|theta|``
    remains within ``tolerance_degrees``. For a disturbance experiment,
    rejection metrics use the samples at and after ``disturbance_time``.
    """

    if not experiment or not controller:
        raise ValueError("experiment and controller names must not be empty")
    if not isfinite(tolerance_degrees) or tolerance_degrees <= 0.0:
        raise ValueError("tolerance_degrees must be positive and finite")
    if len(result.time) == 0:
        raise ValueError("result must contain at least one sample")
    if disturbance_time is not None and not isfinite(disturbance_time):
        raise ValueError("disturbance_time must be finite")

    time = result.time
    angle_degrees = np.rad2deg(result.state_history[:, 0])
    absolute_angle = np.abs(angle_degrees)
    initial_deviation = absolute_angle[0]
    peak_angle = float(np.max(absolute_angle))
    settling_time = _settling_time(
        time,
        angle_degrees,
        tolerance_degrees,
    )

    termination_reason = result.termination_reason or ""
    if disturbance_time is None:
        post_disturbance_angle = absolute_angle
        recovery_time = float("nan")
        energy_reference_index = 0
    else:
        post_disturbance = time >= disturbance_time
        if not np.any(post_disturbance):
            if not termination_reason:
                raise ValueError(
                    "disturbance_time must be within the result time"
                )
            # A fallover before the event has no recovery interval. Keep the
            # available trajectory measurable instead of rejecting the run.
            post_disturbance_angle = absolute_angle
            recovery_time = float("nan")
            energy_reference_index = 0
        else:
            energy_reference_index = int(np.flatnonzero(post_disturbance)[0])
            post_disturbance_angle = absolute_angle[post_disturbance]
            post_settling = _settling_time(
                time,
                angle_degrees,
                tolerance_degrees,
                disturbance_time,
            )
            recovery_time = (
                float(post_settling - disturbance_time)
                if isfinite(post_settling)
                else float("nan")
            )

    torque = result.control_input
    interval_duration = np.diff(time)
    held_torque = torque[:-1]
    simulated_duration = float(np.sum(interval_duration))
    if simulated_duration <= 0.0:
        raise ValueError("result must span a positive duration")
    # The simulator holds held_torque[i] over interval_duration[i]. The final
    # control sample is a repeated endpoint and must not receive weight.
    integral_abs_torque = float(
        np.sum(np.abs(held_torque) * interval_duration)
    )
    rms_torque = float(
        sqrt(np.sum(held_torque**2 * interval_duration) / simulated_duration)
    )
    energy_reference = float(result.energy[energy_reference_index])
    energy_deviation = result.energy[energy_reference_index:] - energy_reference
    energy_interval_duration = np.diff(time[energy_reference_index:])
    if energy_interval_duration.size:
        energy_deviation_rms = float(
            sqrt(
                np.sum(
                    energy_deviation[:-1] ** 2 * energy_interval_duration
                )
                / np.sum(energy_interval_duration)
            )
        )
    else:
        energy_deviation_rms = 0.0
    overshoot = max(0.0, peak_angle - initial_deviation)
    settling_success = isfinite(settling_time)
    rejection_success = disturbance_time is None or isfinite(recovery_time)
    success = int(
        not termination_reason
        and settling_success
        and rejection_success
    )
    termination_time = (
        float(time[-1]) if termination_reason else float("nan")
    )
    fallover_penalty = float(termination_reason == "angle_limit")

    return ExperimentMetrics(
        experiment=experiment,
        controller=controller,
        simulated_duration_s=simulated_duration,
        settling_time_s=settling_time,
        rmse_angle_deg=float(sqrt(np.mean(angle_degrees**2))),
        peak_angle_deg=peak_angle,
        overshoot_deg=overshoot,
        max_deviation_deg=float(np.max(post_disturbance_angle)),
        recovery_time_s=recovery_time,
        peak_torque_nm=float(np.max(np.abs(torque))),
        rms_torque_nm=rms_torque,
        mean_abs_torque_nm=integral_abs_torque / simulated_duration,
        integral_abs_torque_nm_s=integral_abs_torque,
        final_angle_deg=float(angle_degrees[-1]),
        wheel_drift_m=float(abs(result.state_history[-1, 2])),
        final_energy_j=float(result.energy[-1]),
        success=success,
        termination_reason=termination_reason,
        termination_time_s=termination_time,
        fallover_penalty=fallover_penalty,
        energy_reference_j=energy_reference,
        energy_deviation_rms_j=energy_deviation_rms,
        final_energy_deviation_j=float(
            abs(result.energy[-1] - energy_reference)
        ),
    )


__all__ = ["ExperimentMetrics", "compute_metrics"]
