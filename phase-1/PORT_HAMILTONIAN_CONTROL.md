# Port-Hamiltonian Control

## Concept

The robot is treated as an energy system rather than only an error
tracking system.

## Dynamics

xdot = (J-R) grad H(x) + Gu

Where:

J: Energy exchange structure

R: Energy dissipation

G: Input mapping

The plant Hamiltonian uses the physical gravitational potential
`V_phys = m g l cos(theta)`. The controller uses a separate upright-relative
shaped potential `V_c = m g l (1 - cos(theta))` for the energy error.

## Controller

Energy shaping:

u = -Ke(H-Hd)-Kd\*q_dot

The controller: - regulates energy - injects damping - attempts to stabilize
the equilibrium

Validation boundary: the scalar law is an energy-control reference, not a
complete local stabilizer for the inverted plant. At a nonzero lean with zero
velocity, `H-Hd` has no lean-direction information. A practical stabilizing
extension needs a signed potential-gradient or local angle-feedback term.
