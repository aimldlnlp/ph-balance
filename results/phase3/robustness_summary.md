# Phase 3 robustness summary

Successful-run means exclude terminated runs; fallover counts and success rates are reported separately.

| group | controller | scenario_count | success_count | success_rate | fallover_count | mean_rmse_successful_deg | mean_settling_successful_s | mean_wheel_drift_successful_m | mean_effort_successful_nm_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | PID | 3 | 3 | 1.000000 | 0 | 1.562124 | 0.146667 | 0.630559 | 0.327699 |
| main | PH-V1 | 3 | 0 | 0.000000 | 3 | - | - | - | - |
| main | PH-V2 | 3 | 0 | 0.000000 | 3 | - | - | - | - |
| main | PH-V3 | 3 | 1 | 0.333333 | 2 | 3.398769 | 1.420000 | 0.659088 | 0.442971 |
| main | Hybrid PH | 3 | 3 | 1.000000 | 0 | 2.826206 | 3.433333 | 0.052710 | 0.341744 |
| angle_sweep | PID | 4 | 4 | 1.000000 | 0 | 4.329944 | 0.260000 | 1.016400 | 0.557054 |
| angle_sweep | PH-V1 | 4 | 0 | 0.000000 | 4 | - | - | - | - |
| angle_sweep | PH-V2 | 4 | 0 | 0.000000 | 4 | - | - | - | - |
| angle_sweep | PH-V3 | 4 | 1 | 0.250000 | 2 | 3.398769 | 1.420000 | 0.659088 | 0.442971 |
| angle_sweep | Hybrid PH | 4 | 2 | 0.500000 | 0 | 2.081297 | 0.940000 | 0.013432 | 0.197908 |
| disturbance_sweep | PID | 5 | 5 | 1.000000 | 0 | 0.183627 | 1.052000 | 0.953224 | 0.192010 |
| disturbance_sweep | PH-V1 | 5 | 0 | 0.000000 | 5 | - | - | - | - |
| disturbance_sweep | PH-V2 | 5 | 0 | 0.000000 | 5 | - | - | - | - |
| disturbance_sweep | PH-V3 | 5 | 1 | 0.200000 | 4 | 0.850023 | 5.600000 | 0.301448 | 0.235299 |
| disturbance_sweep | Hybrid PH | 5 | 5 | 1.000000 | 0 | 2.195256 | 5.692000 | 0.164558 | 0.411133 |
