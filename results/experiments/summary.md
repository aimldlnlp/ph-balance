| Experiment | Controller | Duration (s) | Settling (s) | RMSE (deg) | Max deviation (deg) | Recovery (s) | Peak torque (N m) | Mean abs torque (N m) | Energy deviation RMS (J) | Success | Termination time (s) | Fallover penalty | Termination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal_stabilization | PID | 4.000 | 0.220 | 2.250 | 15.000 | - | 3.000 | 0.101 | 0.207 | 1 | - | 0.000 | - |
| nominal_stabilization | Port-Hamiltonian | 0.741 | - | 42.493 | 85.000 | - | 0.187 | 0.134 | 0.019 | 0 | 0.741 | 1.000 | angle_limit |
| external_disturbance | PID | 8.000 | 0.000 | 0.177 | 1.252 | 0.000 | 1.024 | 0.023 | 0.452 | 1 | - | 0.000 | - |
| external_disturbance | Port-Hamiltonian | 5.762 | - | 14.517 | 85.000 | - | 0.213 | 0.018 | 0.087 | 0 | 5.762 | 1.000 | angle_limit |
| parameter_uncertainty | PID | 4.000 | 0.220 | 2.260 | 15.000 | - | 3.000 | 0.099 | 0.232 | 1 | - | 0.000 | - |
| parameter_uncertainty | Port-Hamiltonian | 0.691 | - | 41.759 | 85.000 | - | 0.224 | 0.161 | 0.033 | 0 | 0.691 | 1.000 | angle_limit |
