# Experiment Plan

## Experiment 1: Stabilization

Initial condition:

theta = 15 degrees

Measure: - settling time - RMSE - overshoot

## Experiment 2: Disturbance Rejection

Apply external torque disturbance.

The reproducible run uses a 1 N m body-torque pulse from 5.0 s to 5.2 s.

Measure: - maximum deviation - recovery time

## Experiment 3: Parameter Uncertainty

Change: - robot mass - friction coefficient

Evaluate robustness.

The reproducible uncertainty case increases body mass by 25% and doubles both
viscous friction coefficients while controllers retain nominal parameters.

All three runs stop at |theta| = 85 degrees. This fallover bound prevents the
local upright model from being interpreted as a full-rotation robot model.

## Required Outputs

-   plots
-   CSV metrics
-   animation
-   video
