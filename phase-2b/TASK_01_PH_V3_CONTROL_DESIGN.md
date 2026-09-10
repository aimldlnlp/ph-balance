# Task 01: PH-V3 Control Design

## Role

Act as nonlinear control researcher.

Do not code yet.

## Problem

PH-V2 controls:

-   energy
-   body angle

but does not regulate:

-   wheel position
-   translational drift

## Design Target

Develop:

u = energy term + angle shaping + position shaping + damping

Candidate structure:

tau = -Ke(H-Hd)y + Ktheta*sin(theta) - Ks*s - Kv\*s_dot

## Analyze

Before implementation explain:

1.  Why PH-V2 failed.
2.  Why wheel position regulation is required.
3.  Sign convention with existing motor torque.
4.  Local linear stability expectation.
5.  Physical meaning of each gain.

Wait for approval before coding.
