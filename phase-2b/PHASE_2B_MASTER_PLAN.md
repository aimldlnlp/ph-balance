# PH-Balance Phase 2B Master Plan: PH-V3 Stabilizing Redesign

## Objective

Redesign the Port-Hamiltonian controller after Phase 2 audit showed that
PH-V2 directional shaping was insufficient.

## Current Findings

PH-V1: - Energy-only controller - Failed because energy lacks direction
information

PH-V2: - Added angle shaping - Correct torque sign - Still unstable

Root causes: - Underactuated system - Wheel position mode is
uncontrolled - Translational dynamics remain coupled with body
stabilization

## Phase 2B Goal

Create PH-V3:

Energy shaping + Angle stabilization + Wheel position stabilization +
Damping injection

Target:

A locally stabilizing controller for the self-balancing robot.

## Preserve

Keep:

-   PID baseline
-   PH-V1
-   PH-V2
-   existing experiments
-   existing architecture

PH-V3 becomes the new research candidate.
