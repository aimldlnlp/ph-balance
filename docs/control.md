# Control and energy model

The physical model is a six-degree-of-freedom floating-base Upkie. Roll, yaw,
slip, and contact dynamics remain free. A rolling reduction is used only to
design a local controller. It does not constrain the simulated base.

## Balance and posture

A multibody LQR stabilizer regulates forward position, lean, heading, and four
leg coordinates. The final controller also uses six bounded integral states.
Integration stops when torque clipping or command slew would prevent the
requested correction. Hip and knee support is part of this local multibody
control; PID is not the primary balancing law.

Eight nominal postures supply mass matrices, equilibrium lean, gravity support
and stabilizer matrices. Interpolation follows the height command. The legs
complete three down/up cycles of 4 cm during each 60-second run.

## Energy shaping

The PH term uses the nominal shaped-energy scalar

```text
H_s = m g l (1 - cos(theta)) + 0.5 v^T M v + 10 e_leg^T e_leg
```

Here theta is lean relative to scheduled trim, v contains seven nominal
velocity coordinates, and e_leg is the four-joint posture error. The PH wheel
correction is proportional to -(0.1 H_s + 0.02) times the actuator port
velocity in the final study. It supplements the local stabilizer and is
subject to the same source limits and command slew cap. This combination
is not a proof of global nonlinear stability or passivity.

The energy scene in the demo evaluates this formula offline on saved native
states. It is not an exact log of the noisy observer's internal energy or
the robot's total mechanical energy. The surface is a conditional slice in
lean and body pitch rate. All other coordinates stay at their current values,
and cross terms are retained. The state trail is projected onto that current
slice; the lower curve alone shows energy through time. Energy increases
during disturbances and posture changes are retained.

## Observation and actuation

Timestamped fused attitude, body gyro and joint encoders feed a fixed-lag
nominal predictor. Late packets can be discarded; future packets are never
assimilated early. A 0.2-second supported sensor warmup precedes release.
No fixture acts afterward.

Each wheel is limited to 1.7 Nm and 111 rad/s. Hip and knee motors are limited
to 16 Nm and 28.8 rad/s, with source joint ranges enforced. Commands change
by at most 50 Nm/s. First-order motor lag applies before physical actuation.
Source collision geometry stays enabled throughout.
