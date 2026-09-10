# disturbance_rejection

| Experiment | Controller | Duration (s) | Settling (s) | RMSE angle (deg) | Overshoot (deg) | Max deviation (deg) | Wheel drift (m) | Recovery (s) | Energy deviation RMS (J) | Final energy deviation (J) | Peak torque (N m) | ZOH torque effort (N m s) | Success | Termination time (s) | Fallover penalty | Termination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| disturbance_rejection | PID | 8.000000 | 0.000000 | 0.176552 | 1.252456 | 1.252456 | 0.916494 | 0.000000 | 0.451535 | 0.457853 | 1.024288 | 0.184610 | 1 | - | 0.000000 | - |
| disturbance_rejection | PH-V1 | 5.762325 | - | 14.516535 | 85.000000 | 85.000000 | 0.065837 | - | 0.086611 | 0.055143 | 0.213399 | 0.104727 | 0 | 5.762325 | 1.000000 | angle_limit |
| disturbance_rejection | PH-V2 | 5.793828 | - | 14.268777 | 85.000000 | 85.000000 | 0.096568 | - | 0.116833 | 0.163872 | 0.253675 | 0.142966 | 0 | 5.793828 | 1.000000 | angle_limit |
| disturbance_rejection | PH-V3 | 6.104913 | - | 15.276120 | 85.000000 | 85.000000 | 0.429121 | - | 1.110411 | 1.684088 | 0.843554 | 0.312719 | 0 | 6.104913 | 1.000000 | angle_limit |
| disturbance_rejection | Hybrid PH | 8.000000 | 6.980000 | 2.079380 | 5.790618 | 5.790618 | 0.122069 | 1.980000 | 0.208999 | 0.055478 | 0.968851 | 0.400746 | 1 | - | 0.000000 | - |
