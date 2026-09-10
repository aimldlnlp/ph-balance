# Task 04: Phase 2 Experiments

## Controllers

Compare:

-   PID
-   Port-Hamiltonian V1
-   Port-Hamiltonian V2

## Experiment 1: Stabilization

Initial condition:

theta = 15 degrees

Measure:

-   settling time
-   RMSE
-   overshoot
-   torque effort

## Experiment 2: Disturbance Rejection

Apply:

1 Nm body torque pulse

Duration:

5.0s - 5.2s

Measure:

-   maximum deviation
-   recovery time
-   energy convergence

## Experiment 3: Parameter Uncertainty

Change:

-   mass +25%
-   friction x2

Measure:

-   stability
-   RMSE change
-   control effort

## Additional Experiments

Angle sweep:

5, 15, 30, 45 degrees

Disturbance sweep:

0.2, 0.5, 1.0, 1.5, 2.0 Nm

## Outputs

Create:

results/phase2/

with:

-   plots
-   CSV metrics
-   summary report
