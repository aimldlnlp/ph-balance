# PH-Balance Phase 3: Hybrid Port-Hamiltonian Control

## Objective

Develop a hybrid controller combining Port-Hamiltonian energy shaping
with a robust local stabilizer.

Phase 2 findings: - PH-V1 failed because energy lacked direction
information. - PH-V2 improved angle handling but lacked wheel mode
control. - PH-V3 achieved local stability but lacked nonlinear
robustness.

## Research Question

Can a hybrid PH controller preserve energy-based interpretation while
achieving practical robustness?

## Architecture

u = u_PH + u_stabilizer

u_PH: - energy shaping - damping injection

u_stabilizer: - LQR / state feedback / local stabilizer

Keep PID, PH-V1, PH-V2, PH-V3 as baselines.
