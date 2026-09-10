# Task 02: Implement Port-Hamiltonian V2

## Target File

src/controllers/port_hamiltonian.py

## Requirements

Do not rewrite architecture.

Preserve compatibility with simulator.

Keep PH-V1 available.

Implement PH-V2 with:

-   energy_gain Ke
-   angle_gain Ktheta
-   damping_gain Kd
-   desired_energy Hd
-   torque_limit

## Controller Interface

Maintain existing controller interface.

## Code Quality

Follow AGENTS.md:

-   type hints
-   docstrings
-   readable implementation
-   explicit assumptions

## Testing

Add tests for:

-   controller output direction
-   zero equilibrium behavior
-   torque saturation
-   parameter configuration

Do not tune gains yet.
