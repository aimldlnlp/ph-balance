"""Integrated floating-base pilot. Held-out seed evaluation is a separate step."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from src.simulation.upkie import ROOT, TIMESTEP, CONTROL_PERIOD, WHEEL_RADIUS
from src.simulation.upkie_3d import build_floating, floating_lqr, leg_pose
from src.simulation.upkie_observer import Sensors, Observer
from src.simulation.upkie_active_lqr import active_design, ActiveObserver


@dataclass(frozen=True)
class Case:
    seed: int = 1000
    duration: float = 60.
    pitch: float = 10.
    heading: float = 15.
    torso_scale: float = 1.
    friction: float = 0.8
    noise_deg: float = 0.2
    active: bool = False
    pulse: float = 0.68
    lateral: float = 1.


def schedule(active=False, q=(150., 8., 80., 15.), r=0.15):
    table = []
    for knee in (np.linspace(0.5, 1.2, 8) if active else [0.5]):
        robot = build_floating(knee=float(knee), active=active)
        data = robot.initial_data()
        support = np.zeros(robot.model.nv)
        for side in ("left", "right"):
            body = robot.model.body(side+"_wheel_center").id
            jac = np.zeros((3, robot.model.nv))
            mujoco.mj_jac(robot.model, data, jac, None, data.xpos[body], body)
            support += jac[2]*sum(robot.model.body_mass)*9.81/2
        leg_feedforward = (data.qfrc_bias-support)[robot.model.jnt_dofadr[robot.leg_ids]]
        row = [knee, data.xpos[robot.model.body("torso").id, 2], robot.trim,
               robot.length, robot.body_mass, *floating_lqr(robot, q, r), *leg_feedforward]
        if active:
            design = active_design(robot)
            row += [*design["gain"].ravel(), *design["ad"].ravel(), *design["bd"].ravel(), *design["equilibrium"], *design["feedforward"], *design["reduced_mass"].ravel()]
        table.append(row)
    return np.array(table)


def unpack_design(row):
    return dict(gain=row[13:97].reshape(6, 14), ad=row[97:293].reshape(14, 14), bd=row[293:377].reshape(14, 6),
                equilibrium=row[377:391], feedforward=row[391:397], reduced_mass=row[397:446].reshape(7, 7))


def reference(time, table, active):
    high = table[0, 1]
    # Three complete down/up cycles; two-second ramps and six-second plateaus.
    phase = (time-6.)%16.
    fraction = (0.5-0.5*np.cos(np.pi*phase/2) if phase < 2 else
                1. if phase < 8 else 0.5+0.5*np.cos(np.pi*(phase-8)/2) if phase < 10 else 0.)
    height = high-0.04*fraction if active and 6 <= time < 54 else high
    knee = np.interp(height, table[::-1, 1], table[::-1, 0]) if active else 0.5
    return np.array([np.interp(knee, table[:, 0], table[:, i]) for i in range(table.shape[1])]), height


def run(case, table, *, hybrid=True, energy_gain=0.1, damping_gain=0.01):
    if not np.isfinite(list(asdict(case).values())).all() or case.duration <= 0:
        raise ValueError("Invalid case")
    robot = build_floating(active=case.active, torso_scale=case.torso_scale, friction=case.friction)
    model = robot.model
    data = robot.initial_data(case.pitch)
    # Estimator receives only a nominal model, never the perturbed plant model.
    nominal_robot = build_floating(active=case.active)
    _, ad, bd = floating_lqr(nominal_robot, matrices=True)
    sensor = Sensors(robot, case.seed, case.noise_deg)
    observer = ActiveObserver(unpack_design(table[0])) if case.active else Observer(ad, bd, nominal_robot.trim)
    commands = np.zeros(model.nu)
    if case.active:
        commands[2:] = table[0, 9:13]
    records, qpositions, qvelocities = [], [], []
    reason = None
    initial_contacts = robot.nonwheel_ground_contacts(data)
    dofs = model.jnt_dofadr[np.r_[robot.wheel_ids, robot.leg_ids]]
    joints = np.r_[robot.wheel_ids, robot.leg_ids]
    names = [model.joint(int(j)).name for j in joints]
    limits = np.array([robot.limits[n]["velocity"] for n in names])
    max_speed = np.zeros(len(names))
    torso = model.body("torso").id
    stride = round(CONTROL_PERIOD/TIMESTEP)
    estimated = np.zeros(4)
    for step in range(round(case.duration/TIMESTEP)+1):
        time = step*TIMESTEP
        # Ground truth is used only here (sensor emulation), actuation safety,
        # external disturbance application and evaluation below.
        if step%stride == 0:
            sensor.sample(data, time)
            for observation in sensor.receive(time):
                observer.update(observation, time)
            estimate = observer.estimate(time)
            nominal, target_height = reference(time, table, case.active)
            if case.active:
                observer.design = unpack_design(nominal)
            if estimate is not None:
                previous_commands = commands.copy()
                attitude, gyro, xy, speed = estimate
                yaw = attitude[2]
                position = xy@np.array([np.cos(yaw), np.sin(yaw)])
                lean = attitude[1]-nominal[2]
                estimated = np.array([lean, gyro[1], position, speed])
                torque = -nominal[5:9]@estimated
                if case.active:
                    design = observer.design
                    target_state = design["equilibrium"].copy()
                    target_state[2] = np.deg2rad(case.heading) if time >= 3 else 0.
                    multibody_input = design["feedforward"]-design["gain"]@(observer.active_state(time)-target_state)
                    torque = sum(multibody_input[:2])
                if hybrid:
                    port = speed/WHEEL_RADIUS-gyro[1]
                    energy = nominal[4]*9.81*nominal[3]*(1-np.cos(lean)) + 0.5*nominal[4]*speed**2
                    if case.active:
                        observed_state = observer.active_state(time)
                        velocity = observed_state[7:]
                        port += (sum(velocity[3:5])-sum(velocity[5:7]))/2
                        posture_error = observed_state[3:7]-design["equilibrium"][3:7]
                        energy = nominal[4]*9.81*nominal[3]*(1-np.cos(lean)) + .5*velocity@design["reduced_mass"]@velocity + 10*posture_error@posture_error
                    torque -= (energy_gain*energy+damping_gain)*port
                heading_target = np.deg2rad(case.heading) if time >= 3 else 0.
                yaw_error = (heading_target-yaw+np.pi)%(2*np.pi)-np.pi
                turn = np.clip(0.7*yaw_error-0.12*gyro[2], -0.5, 0.5)
                commands[:2] = [0.5*torque-turn, 0.5*torque+turn]
                if case.active:
                    commands[:] = multibody_input
                    commands[:2] += .5*(torque-sum(multibody_input[:2]))
                commands = np.clip(commands, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                if case.active:
                    commands = np.clip(commands, previous_commands-50*CONTROL_PERIOD, previous_commands+50*CONTROL_PERIOD)
            observer.command(time, commands if case.active else sum(commands[:2]))
        roll, pitch, yaw_true = robot.orientation(data)
        contacts = robot.nonwheel_ground_contacts(data)
        max_speed = np.maximum(max_speed, abs(data.qvel[dofs]))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason = "nonfinite"
        elif contacts:
            reason = "nonwheel_ground_contact:"+",".join(contacts)
        elif np.any(max_speed > limits+1e-6):
            reason = "joint_speed_limit"
        elif any("lower" in robot.limits[n] and not robot.limits[n]["lower"]-1e-4 <= data.qpos[model.jnt_qposadr[j]] <= robot.limits[n]["upper"]+1e-4 for n, j in zip(names, joints)):
            reason = "joint_position_limit"
        elif abs(pitch) > np.deg2rad(70) or abs(roll) > np.deg2rad(45):
            reason = "attitude_limit"
        if step%stride == 0 or reason or step == round(case.duration/TIMESTEP):
            centers = np.mean([data.xpos[model.body(s+"_wheel_center").id] for s in ("left", "right")], axis=0)
            records.append([time, roll, pitch-nominal[2], yaw_true, *centers[:2], data.xpos[torso, 2], target_height,
                            *estimated, *(commands.tolist()+[0.]*(6-len(commands))),
                            -1. if observer.last is None else observer.last.sampled_at])
            qpositions.append(data.qpos.copy())
            qvelocities.append(data.qvel.copy())
        if reason or step == round(case.duration/TIMESTEP):
            break
        data.ctrl[:] = commands
        data.xfrc_applied[:] = 0.
        if 15 <= time < 15.2:
            data.xfrc_applied[torso, 4] = case.pulse
        if 35 <= time < 35.2:
            data.xfrc_applied[torso, 4] = -case.pulse
        if 25 <= time < 25.2:
            data.xfrc_applied[torso, 1] = case.lateral
        mujoco.mj_step(model, data)
    history = np.array(records)
    metrics = measure(history, case, reason)
    metrics.update(initial_nonwheel_contacts=initial_contacts, peak_joint_speed=max_speed.tolist())
    robot.sensor_history = sensor.history
    return robot, history, np.array(qpositions), np.array(qvelocities), metrics


COLUMNS = "time roll lean yaw x y height height_target lean_est rate_est position_est speed_est left_torque right_torque left_hip_torque left_knee_torque right_hip_torque right_knee_torque sampled_at".split()


def measure(h, case, reason):
    time = h[:, 0]
    events = [0., 15.2, 35.2]
    settling = []
    for start, end in zip(events, [15., 35., case.duration]):
        indices = np.flatnonzero((time >= start)&(time <= end))
        invalid = indices[abs(h[indices, 2]) > np.deg2rad(2)]
        t = time[invalid[-1]]+CONTROL_PERIOD if len(invalid) else start
        settling.append(None if end-t < 5 or time[-1] < end-1e-8 else float(t-start))
    excursion = np.linalg.norm(h[:, 4:6]-h[0, 4:6], axis=1)
    final = float(excursion[-1])
    heading = float(abs(np.rad2deg((h[-1, 3]-np.deg2rad(case.heading)+np.pi)%(2*np.pi)-np.pi)))
    complete = reason is None and abs(time[-1]-case.duration) < 1e-8
    # Guardrails are evaluated on true physical state, never the estimate.
    settled = (time >= 3)
    for event in (15.2, 35.2):
        settled &= ~((time >= event-0.2)&(time < event+3))
    if case.active:
        for event in (6, 14, 22, 30, 38, 46):
            settled &= ~((time >= event)&(time < event+4))
    peak_lean = float(np.max(abs(np.rad2deg(h[settled, 2])))) if np.any(settled) else None
    height_error = float(np.max(abs(h[settled, 6]-h[settled, 7]))) if np.any(settled) else None
    robust = complete and all(s is not None and s <= 3 for s in settling) and final <= .15 and max(excursion) <= .5
    passed = robust and heading <= 5 and (not case.active or (peak_lean is not None and peak_lean <= 2 and height_error <= .01))
    gates = dict(completed=complete, settling=all(s is not None and s <= 3 for s in settling),
                 final_position=final <= .15, excursion=max(excursion) <= .5, heading=heading <= 5)
    if case.active:
        gates.update(lean=peak_lean is not None and peak_lean <= 2, height=height_error is not None and height_error <= .01)
    return dict(case=asdict(case), hybrid=None, completed=bool(complete), failure=reason, success=bool(passed), robust_success=bool(robust),
                failed_metric_gates=[name for name, valid in gates.items() if not valid],
                duration=float(time[-1]), settling_s=settling, final_position_m=final, max_excursion_m=float(max(excursion)),
                heading_error_deg=heading, settled_peak_lean_deg=peak_lean, settled_peak_height_error_m=height_error,
                wheel_absolute_torque_integral=float(np.diff(time)@np.sum(abs(h[:-1, 12:14]), axis=1)),
                all_absolute_torque_integral=float(np.diff(time)@np.sum(abs(h[:-1, 12:18]), axis=1)))


def save_run(directory, case, hybrid, table, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    robot, h, qpos, qvel, metrics = run(case, table, hybrid=hybrid, **kwargs)
    metrics["hybrid"] = hybrid
    np.savez_compressed(directory/"trajectory.npz", history=h, qpos=qpos, qvel=qvel)
    observations = np.array([[o.sampled_at, o.available_at, *o.attitude, *o.gyro,
                              *(o.joint_position.tolist()+[0.]*(6-len(o.joint_position))),
                              *(o.joint_velocity.tolist()+[0.]*(6-len(o.joint_velocity)))] for o in robot.sensor_history])
    np.savez_compressed(directory/"observations.npz", observations=observations)
    np.savetxt(directory/"trajectory.csv", h, delimiter=",", header=",".join(COLUMNS), comments="")
    (directory/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mujoco.mj_saveLastXML(str(directory/"model.xml"), robot.model)
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--duration", type=float, default=60.)
    args = parser.parse_args()
    table = schedule(args.active)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savetxt(output/"nominal_schedule.csv", table, delimiter=",")
    for angle in (-10., 10.):
        case = Case(pitch=angle, heading=15 if angle > 0 else -15, active=args.active, duration=args.duration)
        print(json.dumps(save_run(output/f"pitch_{angle:+g}", case, True, table)), flush=True)
