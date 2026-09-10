"""Physical parameter containers.

Numerical experiment assumptions belong in configuration code. This module
only defines the typed data passed to the model and controllers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotParameters:
    """Physical parameters for the planar two-wheel robot.

    Masses and inertias are per wheel where the field name says so. Units are
    SI: kilograms, kilograms metre squared, metres, and metres per second
    squared. Translational friction is in N s/m and rotational friction is in
    N m s/rad.
    """

    body_mass: float
    wheel_mass_per_wheel: float
    body_inertia: float
    wheel_inertia_per_wheel: float
    center_of_mass_length: float
    wheel_radius: float
    gravity: float
    translational_friction: float
    rotational_friction: float
