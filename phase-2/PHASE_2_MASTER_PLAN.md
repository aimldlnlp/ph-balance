# PH-Balance Phase 2 Master Plan

## Objective

Upgrade the Port-Hamiltonian controller from V1 energy regulation into a
locally stabilizing energy-based controller.

## Current Status

Completed:

-   Physics model validated
-   Hamiltonian validated
-   Simulator validated
-   PID baseline validated
-   Visualization pipeline completed

Problem:

Port-Hamiltonian V1 is mathematically consistent but cannot stabilize
the inverted pendulum.

## Root Cause

The controller only uses energy deviation:

u = -Ke(H-Hd)-Kd\*y

Energy does not contain direction information.

A +15 degree lean and -15 degree lean have similar energy but require
opposite torque.

Therefore the controller requires directional stabilization.

## Phase 2 Goal

Create Port-Hamiltonian V2:

u = energy shaping + potential shaping + damping injection

Target:

-   stabilize 15 degree initial condition
-   reject disturbance
-   handle parameter uncertainty
-   compare against PID and PH-V1

## Success Criteria

Phase 2 succeeds if:

1.  PH-V2 stabilizes the robot.
2.  PH-V2 performs disturbance rejection.
3.  PH-V2 survives parameter uncertainty.
4.  Results are reproducible.
5.  Failure mechanism of PH-V1 is documented.
