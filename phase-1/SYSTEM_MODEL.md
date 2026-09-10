# System Model

## Robot

Two-wheel inverted pendulum robot.

## State

x = \[ theta, theta_dot, wheel_position, wheel_velocity\]

## Input

Motor torque.

## Assumptions

-   Rigid body model
-   Planar motion
-   No wheel slip
-   Constant gravity
-   Viscous friction
-   Ideal commanded motor torque (no motor electrical or actuator dynamics)
-   Local upright model; validity is limited to the configured fallover angle

## Energy

Hamiltonian:

H(q,p)=T(q,p)+V(q)

Kinetic energy:

T = motion energy

Potential energy:

V = gravitational energy

The implementation keeps two explicit potentials: the physical plant
potential `V_phys = m g l cos(theta)` and the upright-relative control-shaped
potential `V_c = m g l (1 - cos(theta))`. `V_phys` is used for plant energy
conservation and `V_c` is used by the reference energy-shaping controller.
