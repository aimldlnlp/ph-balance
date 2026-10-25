# Evaluation

The repository includes fixed case definitions and compact per-episode metrics.
Large native trajectories are produced by the evaluation commands rather than
stored in Git. Every scheduled case counts, including early failures.

## Final stress study

[evaluation/stress](../evaluation/stress) contains 200 new 60-second cases.
PH passes 199/200, with a 95% Wilson interval of 97.22% to 99.91%. The acceptance
gate was at least 190 successes. The only change from the first stress study
was PH damping from 0.01 to 0.02; energy gain stayed at 0.1.

Four damping values were tested on the same eight development cases. The
selected value then passed 31/32 random pilot cases before the final cases
were fixed. Development and final seeds are separate.

Every final case completes 60 seconds without a body/bracket ground contact
or physical-limit termination. Seed 494124 fails the required five-second
settling confirmation after the persistent disturbance and before the next
pitch pulse. Its full duration does not turn it into a success.

Success also requires recovery within 3 seconds, final position within 0.15 m,
excursion within 0.50 m, final heading within 5 degrees, and settled active-leg
lean/height errors within 2 degrees and 1 cm. Reference ramps and disturbance
transients have explicit exclusion windows for the settled-error metrics.
Recovery is checked separately after each disturbance. The source acceptance
functions and protocol give the exact windows.

Uncertainty includes torso mass/inertia +/-20%, friction 0.4 to 0.8, attitude
noise 0.2 to 0.5 degrees RMS, pitch bias +/-0.25 degrees, gyro bias +/-0.1 degrees/s,
and heading drift +/-0.01 degrees/s. Delay is 10 to 25 ms with +/-5 ms jitter
and 10% independent packet loss. Motor lag is 5 to 20 ms. Pitch pulses are
0.4 to 0.8 Nm for 0.2 s in both directions. A 3 N sideways force lasts 0.2 s,
and persistent pitch torque is +/-0.1 Nm after 8 s.

## Matched PH contribution study

[evaluation/paired](../evaluation/paired) contains 200 case pairs, 400 recordings.
PH with integral correction passes 189/200; matched LQI passes 121/200. The
190/200 target was missed. All eleven PH failures involve ankle ground contact.
Both arms share estimator, nominal gains, integral support, limits, scenarios
and transport streams. Only the PH terms differ. This is not a comparison
against the best independently tuned LQI controller.

On the 121 common successful pairs, absolute applied wheel-torque effort falls
14.60% (paired bootstrap 95%: 6.72% to 22.84%), and positive mechanical work
falls 10.94% (5.73% to 16.95%). Cost and success are reported separately. Short
failed exposures are not pooled with successful full minutes. Mechanical
work excludes electrical/thermal losses and does not measure battery use.

## Earlier study

[evaluation/initial](../evaluation/initial) contains 504 cases: PH 200/200,
LQR 200/200, active legs 100/100 and four dedicated 3D cases 4/4. The attitude
noise/delay and disturbance envelope are less demanding than the final study.
PH reduced wheel effort by 1.013% (95% paired bootstrap: 0.956% to 1.070%).
The requested 10% effect was not supported.

## Reproduction and scope

Case protocols include seeds, settings and source hashes. Exact source bytes
are preserved across platforms. Runs save observations, transport decisions,
native qpos/qvel and applied torques. Analysis checks timestamps, exposure,
physical limits and denominators.

This is simulation evidence. Fused heading, warmup and first-order motor errors
are assumptions. Hardware calibration, raw accelerometer fusion, magnetic
disturbances, electrical losses and a global stability proof remain untested.
