"""Shared controller contract."""

from typing import Protocol

from src.models.parameters import RobotParameters
from src.models.state import RobotState


class Controller(Protocol):
    """Interface implemented by torque-producing controllers."""

    def compute_torque(
        self,
        time: float,
        state: RobotState,
        parameters: RobotParameters,
    ) -> float:
        """Return the commanded motor torque in newton metres."""

        ...
