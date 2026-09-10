# Task 01: Port-Hamiltonian Controller Redesign

## Objective

Design the new Port-Hamiltonian V2 controller before implementation.

## Problem

PH-V1:

u = -Ke(H-Hd)-Kd\*y

only regulates energy.

Limitation:

Energy is symmetric and cannot determine lean direction.

## New Controller

Implement the concept:

u = u_energy + u_shape + u_damping

## Energy Shaping

u_energy = -Ke(H-Hd)y

Purpose:

-   regulate energy level
-   move system toward desired equilibrium

## Potential Gradient Stabilization

Add:

u_shape = -Ktheta\*sin(theta)

Purpose:

-   provide restoring torque
-   encode direction information

## Damping Injection

Add:

u_damping = -Kd\*theta_dot

Purpose:

-   reduce oscillation
-   improve convergence

## Required Explanation

Before coding, explain:

-   why PH-V1 failed
-   why potential shaping fixes the issue
-   sign convention
-   expected stability behavior
