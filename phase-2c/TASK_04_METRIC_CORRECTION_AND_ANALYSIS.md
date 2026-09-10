# Task 04: Metric Correction

Fix experiment metrics.

Torque effort: Use zero-order-hold calculation:

sum(abs(u_i))\*dt

Do not use trapezoidal integration.

Energy metric: Use clear terminology.

Avoid: energy convergence

Prefer: energy deviation

Add: - success flag - termination time - fallover penalty
