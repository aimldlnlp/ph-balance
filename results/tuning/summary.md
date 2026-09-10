# Controller tuning

PID was tuned in three staged sweeps: proportional authority, derivative damping, then a small integral term. The selected gains were chosen as a speed/effort trade-off rather than the most aggressive response.

Selected PID: `Kp=30.0`, `Ki=0.5`, `Kd=3.0`.

The Port-Hamiltonian grid used non-negative gains and the same nominal +15 degree rest condition. Feasible stabilizing pairs found: `0`.

At the initial condition, `H>0` and the actuator-port velocity is zero, so the implemented `u=-Ke(H-Hd)-Kd*y` law commands negative torque for every `Ke>0`. The plant requires positive corrective torque for this lean direction. No positive `Ke/Kd` tuning can therefore meet the stabilization objective without changing the controller sign/structure.

The current `Ke=0.1`, `Kd=1.0` pair is retained only as a bounded reference for the comparison plots; it is not reported as a valid stabilizing tune.

Rows terminated by `angle_limit` have shorter horizons; their effort values are diagnostic and must not be interpreted as controller efficiency.

## Selected-gain validation

| Experiment | Controller | Duration (s) | Settling (s) | RMSE (deg) | Recovery (s) | Mean abs torque (N m) | Stable | Termination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| nominal_stabilization | PID | 4.000 | 0.220 | 2.250 | - | 0.093 | 1 | - |
| nominal_stabilization | Port-Hamiltonian | 0.741 | - | 42.493 | - | 0.135 | 0 | angle_limit |
| external_disturbance | PID | 8.000 | 0.000 | 0.177 | 0.000 | 0.023 | 1 | - |
| external_disturbance | Port-Hamiltonian | 5.762 | - | 14.517 | - | 0.018 | 0 | angle_limit |
| parameter_uncertainty | PID | 4.000 | 0.220 | 2.260 | - | 0.092 | 1 | - |
| parameter_uncertainty | Port-Hamiltonian | 0.691 | - | 41.759 | - | 0.162 | 0 | angle_limit |
