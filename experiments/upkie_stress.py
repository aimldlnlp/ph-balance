"""Versioned stress experiments; frozen v1 simulation sources stay unchanged.

The rollout is deliberately separate from the frozen runner so its hash-checked
reproduction commands remain usable. Model, schedule and acceptance are reused.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
from scipy.linalg import logm, solve_discrete_are
from scipy.signal import cont2discrete

from experiments.upkie_3d_pilot import Case, schedule, reference, unpack_design, measure as baseline_measure, COLUMNS
from experiments.audit_upkie_3d_native import coordinates
from experiments.evaluate_upkie_3d import hashes, wilson
from src.simulation.upkie import ROOT, TIMESTEP, CONTROL_PERIOD, WHEEL_RADIUS
from src.simulation.upkie_3d import build_floating, floating_lqr
from src.simulation.upkie_observer import Sensors, Observer
from src.simulation.upkie_active_lqr import ActiveObserver


@dataclass(frozen=True)
class StressCase(Case):
    delay: float = .01
    jitter: float = 0.
    dropout: float = 0.
    pitch_bias_deg: float = 0.
    gyro_bias_deg_s: float = 0.
    heading_drift_deg_s: float = 0.
    motor_tau: float = 0.
    persistent_torque: float = 0.
    persistent_start: float = 8.
    warmup: float = 0.

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Nonfinite stress case")
        if (min(self.delay, self.jitter, self.motor_tau, self.noise_deg, self.persistent_start, self.warmup) < 0
                or not 0 <= self.dropout < 1 or self.duration <= 0
                or self.torso_scale <= 0 or self.friction <= 0):
            raise ValueError("Invalid stress case")


class StressSensors(Sensors):
    def __init__(self, robot, case):
        super().__init__(robot, case.seed, case.noise_deg, case.delay)
        self.case = case
        self.transport = np.random.default_rng(np.random.SeedSequence([case.seed, 52691]))
        self.dropped, self.stale = 0, 0
        self.delivered_at = -np.inf
        self.delivery_log = []

    def sample(self, data, time):
        super().sample(data, time)
        observation = self.pending.pop()
        c = self.case
        observation = replace(observation,
            available_at=time+max(0., c.delay+self.transport.uniform(-c.jitter, c.jitter)),
            attitude=observation.attitude+np.deg2rad([0., c.pitch_bias_deg, c.heading_drift_deg_s*time]),
            gyro=observation.gyro+np.deg2rad([0., c.gyro_bias_deg_s, c.heading_drift_deg_s]))
        self.history[-1] = observation
        dropped = self.transport.random() < c.dropout
        self.delivery_log.append([time, observation.available_at, float(dropped)])
        if dropped:
            self.dropped += 1
        else:
            self.pending.append(observation)

    def receive(self, time):
        # Late packets can overtake one another. Never pass stale data to observers.
        ready = sorted((o for o in self.pending if o.available_at <= time+1e-12),
                       key=lambda o: o.sampled_at)
        self.pending = type(self.pending)(o for o in self.pending if o.available_at > time+1e-12)
        result = []
        for o in ready:
            if o.sampled_at <= self.delivered_at:
                self.stale += 1
            else:
                result.append(o)
                self.delivered_at = o.sampled_at
        return result


class GapObserver(Observer):
    def update(self, observation, now):
        if observation.available_at > now+1e-12 or observation.sampled_at > now:
            raise ValueError("Future observation")
        if self.last is not None:
            gap = round((observation.sampled_at-self.last.sampled_at)/CONTROL_PERIOD)
            if gap <= 0:
                raise ValueError("Out-of-order observation")
            for k in range(gap-1):
                at = self.last.sampled_at+k*CONTROL_PERIOD
                self.sagittal = self.ad@self.sagittal+self.bd*self.commands.get(round(at, 6), 0.)
            # Base update supplies the final step; move its command lookup to that step.
            previous = self.last
            self.attitude += (gap-1)*CONTROL_PERIOD*observation.gyro
            self.last = replace(previous, sampled_at=observation.sampled_at-CONTROL_PERIOD)
            super().update(observation, now)
        else:
            super().update(observation, now)


class GapActiveObserver(ActiveObserver):
    def update(self, observation, now):
        if observation.available_at > now+1e-12 or observation.sampled_at > now:
            raise ValueError("Future observation")
        if self.last is not None:
            gap = round((observation.sampled_at-self.last.sampled_at)/CONTROL_PERIOD)
            if gap <= 0:
                raise ValueError("Out-of-order observation")
            for k in range(gap-1):
                self.state = self.predict(self.state, self.last.sampled_at+k*CONTROL_PERIOD)
            self.last = replace(self.last, sampled_at=observation.sampled_at-CONTROL_PERIOD)
        super().update(observation, now)


def actuator_schedule(table, tau=.01):
    """Augment the nominal local plant with six physical first-order motors."""
    result = []
    for row in table:
        d = unpack_design(row)
        continuous = np.real_if_close(logm(np.block([[d['ad'], d['bd']],
            [np.zeros((6, 14)), np.eye(6)]]))/CONTROL_PERIOD)
        if np.iscomplexobj(continuous):
            raise ValueError('Nominal model has no real continuous realization')
        ac = np.block([[continuous[:14, :14], continuous[:14, 14:]],
                       [np.zeros((6, 14)), -np.eye(6)/tau]])
        bc = np.vstack([np.zeros((14, 6)), np.eye(6)/tau])
        ad, bd, *_ = cont2discrete((ac, bc, np.eye(20), np.zeros((20, 6))), CONTROL_PERIOD)
        q = np.diag([80.,150.,20.,100.,100.,100.,100.,15.,8.,1.,1.,1.,1.,1.]+[0.]*6)
        r = np.diag([.3,.3,.04,.04,.04,.04])
        p = solve_discrete_are(ad, bd, q, r)
        gain = np.linalg.solve(r+bd.T@p@bd, bd.T@p@ad)
        if np.max(abs(np.linalg.eigvals(ad-bd@gain))) >= 1:
            raise ValueError('Augmented local controller is unstable')
        result.append(np.r_[row, gain.ravel(), ad.ravel(), bd.ravel(), tau])
    return np.array(result)


def soft_schedule(table, scale=100.):
    """Lower local bandwidth to retain actuator-lag margin; no plant changes."""
    result = table.copy()
    q = np.diag([80.,150.,20.,100.,100.,100.,100.,15.,8.,1.,1.,1.,1.,1.])
    r = scale*np.diag([.3,.3,.04,.04,.04,.04])
    for row in result:
        d = unpack_design(row)
        p = solve_discrete_are(d['ad'], d['bd'], q, r)
        gain = np.linalg.solve(r+d['bd'].T@p@d['bd'], d['bd'].T@p@d['ad'])
        if np.max(abs(np.linalg.eigvals(d['ad']-d['bd']@gain))) >= 1:
            raise ValueError('Retuned local controller is unstable')
        row[13:97] = gain.ravel()
    return result


INTEGRATED_COORDINATES = [0, 2, 3, 4, 5, 6]


def integral_schedule(table, scale=100.):
    """Integral assistance on six independent regulated coordinates, not pitch."""
    result = []
    c = np.eye(14)[INTEGRATED_COORDINATES]
    for row in soft_schedule(table, scale):
        d = unpack_design(row)
        ad = np.block([[d['ad'], np.zeros((14,6))], [-CONTROL_PERIOD*c, np.eye(6)]])
        bd = np.vstack([d['bd'], np.zeros((6,6))])
        q = np.diag([80.,150.,20.,100.,100.,100.,100.,15.,8.,1.,1.,1.,1.,1.,100.,20.,100.,100.,100.,100.])
        r = scale*np.diag([.3,.3,.04,.04,.04,.04])
        p = solve_discrete_are(ad,bd,q,r)
        gain = np.linalg.solve(r+bd.T@p@bd,bd.T@p@ad)
        if np.max(abs(np.linalg.eigvals(ad-bd@gain))) >= 1:
            raise ValueError('Local integral-assisted controller is unstable')
        result.append(np.r_[row,gain.ravel()])
    return np.array(result)


def actuator_design(row):
    return dict(gain=row[446:566].reshape(6,20), ad=row[566:966].reshape(20,20),
                bd=row[966:1086].reshape(20,6), tau=row[1086])


class ActuatorObserver(GapActiveObserver):
    def __init__(self, design, motor_design):
        super().__init__(design)
        self.motor_design = motor_design
        self.motor_states = {0.: np.zeros(6)}

    def command(self, time, torque):
        super().command(time, torque)
        alpha = np.exp(-CONTROL_PERIOD/self.motor_design['tau'])
        self.motor_states[round(time+CONTROL_PERIOD, 6)] = alpha*self.motor_states[round(time,6)]+(1-alpha)*torque

    def predict(self, state, time):
        d, a = self.design, self.motor_design
        motor = self.motor_states[round(time,6)]
        augmented = np.r_[state-d['equilibrium'], motor-d['feedforward']]
        command = self.commands.get(round(time,6), d['feedforward'])
        return d['equilibrium']+(a['ad']@augmented+a['bd']@(command-d['feedforward']))[:14]


def fast_reference(time, table, active):
    if not active:
        return table[0], table[0, 1]
    phase = (time-6.) % 16.
    fraction = (.5-.5*np.cos(np.pi*phase/2) if phase < 2 else
                1. if phase < 8 else .5+.5*np.cos(np.pi*(phase-8)/2) if phase < 10 else 0.)
    height = table[0, 1]-.04*fraction if 6 <= time < 54 else table[0, 1]
    knee = np.interp(height, table[::-1, 1], table[::-1, 0])
    index = min(max(int(np.searchsorted(table[:, 0], knee))-1, 0), len(table)-2)
    weight = (knee-table[index, 0])/(table[index+1, 0]-table[index, 0])
    return table[index]+weight*(table[index+1]-table[index]), height


def measure(h, case, reason):
    result = baseline_measure(h, case, reason)
    t = h[:, 0]
    events = [(0., 0.)]
    events += [(s, s+.2) for s in (15., 25., 35.) if s < case.duration]
    if case.persistent_torque and case.persistent_start < case.duration:
        events.append((case.persistent_start, case.persistent_start))
    events.sort()
    settling = []
    settled = t >= 3.
    for index, (start, end) in enumerate(events):
        next_start = events[index+1][0] if index+1 < len(events) else case.duration
        before_next = t < next_start-1e-10 if index+1 < len(events) else t <= next_start+1e-10
        selected = np.flatnonzero((t >= end)&before_next)
        bad = selected[abs(h[selected, 2]) > np.deg2rad(2)]
        recovered = t[bad[-1]]+CONTROL_PERIOD if len(bad) else end
        settling.append(None if next_start-recovered < 5.-1e-8 or t[-1] < next_start-1e-8
                        else float(recovered-end))
        settled &= ~((t >= start)&(t < end+3.))
    if case.active:
        for start in (6, 14, 22, 30, 38, 46):
            settled &= ~((t >= start)&(t < start+4))
    lean = float(np.max(abs(np.rad2deg(h[settled, 2])))) if np.any(settled) else None
    height = float(np.max(abs(h[settled, 6]-h[settled, 7]))) if np.any(settled) else None
    gates = dict(completed=result['completed'], settling=all(s is not None and s <= 3 for s in settling),
                 final_position=result['final_position_m'] <= .15, excursion=result['max_excursion_m'] <= .5,
                 heading=result['heading_error_deg'] <= 5)
    if case.active:
        gates.update(lean=lean is not None and lean <= 2, height=height is not None and height <= .01)
    result.update(settling_s=settling, settled_peak_lean_deg=lean, settled_peak_height_error_m=height,
                  failed_metric_gates=[k for k,v in gates.items() if not v], success=bool(all(gates.values())),
                  robust_success=bool(all(gates[k] for k in ('completed','settling','final_position','excursion'))))
    return result



def run(case, table, *, hybrid=True, revised=False, integral_assisted=False, actuator_aware=False, rotor_damping=0., lead_s=0., integral_gain=0., energy_gain=0.1, damping_gain=0.01):
    if not np.isfinite(list(asdict(case).values())).all() or case.duration <= 0:
        raise ValueError("Invalid case")
    robot = build_floating(active=case.active, torso_scale=case.torso_scale, friction=case.friction)
    model = robot.model
    data = robot.initial_data(case.pitch)
    # Estimator receives only a nominal model, never the perturbed plant model.
    nominal_robot = build_floating(active=case.active)
    _, ad, bd = floating_lqr(nominal_robot, matrices=True)
    sensor = StressSensors(robot, case)
    observer = (GapActiveObserver if revised else ActiveObserver)(unpack_design(table[0])) if case.active else (GapObserver if revised else Observer)(ad, bd, nominal_robot.trim)
    if actuator_aware:
        if not case.active or table.shape[1] != 1087:
            raise ValueError('Actuator-aware controller requires active augmented schedule')
        observer = ActuatorObserver(unpack_design(table[0]), actuator_design(table[0]))
    commands = np.zeros(model.nu)
    if case.active:
        commands[2:] = table[0, 9:13]
    records, qpositions, qvelocities, actual_commands = [], [], [], []
    applied = np.zeros(model.nu)
    integral = 0.
    signed_work = np.zeros(model.nu)
    positive_work = np.zeros(model.nu)
    actual_effort = np.zeros(model.nu)
    saturated_steps = 0
    previous_estimate_time = None
    max_estimate_age = 0.
    integrated_error = np.zeros(6)
    previous_saturated = False
    if case.warmup:
        # A supported robot streams sensors before release; no fixture acts after t=0.
        for at in np.arange(-case.warmup, 0., CONTROL_PERIOD):
            sensor.sample(data, float(at))
        available = sensor.receive(0.)
        if available:
            observer.update(available[-1], 0.)
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
            nominal, target_height = fast_reference(time, table, case.active)
            if case.active:
                observer.design = unpack_design(nominal)
                if actuator_aware:
                    observer.motor_design = actuator_design(nominal)
            if estimate is not None:
                previous_commands = commands.copy()
                attitude, gyro, xy, speed = estimate
                max_estimate_age = max(max_estimate_age, time-observer.last.sampled_at)
                yaw = attitude[2]
                position = xy@np.array([np.cos(yaw), np.sin(yaw)])
                lean = attitude[1]-nominal[2]
                estimated = np.array([lean, gyro[1], position, speed])
                control_state = observer.active_state(time) if case.active else estimated.copy()
                if lead_s:
                    # Causal actuator-lag anticipation holds the last issued input.
                    design = observer.design if case.active else None
                    for _ in range(round(lead_s/CONTROL_PERIOD)):
                        control_state = (design['equilibrium']+design['ad']@(control_state-design['equilibrium'])+
                            design['bd']@(commands-design['feedforward'])) if case.active else ad@control_state+bd*sum(commands[:2])
                    if case.active:
                        lean, gyro, position, speed = control_state[1]-nominal[2], gyro.copy(), control_state[0], control_state[7]
                        gyro[1] = control_state[8]
                    else:
                        lean, gyro, position, speed = control_state[0], gyro.copy(), control_state[2], control_state[3]
                        gyro[1] = control_state[1]
                torque = -nominal[5:9]@estimated
                if lead_s and not case.active:
                    torque = -nominal[5:9]@control_state
                if case.active:
                    design = observer.design
                    target_state = design["equilibrium"].copy()
                    target_state[2] = np.deg2rad(case.heading) if time >= 3 else 0.
                    multibody_input = design["feedforward"]-design["gain"]@(control_state-target_state)
                    if integral_assisted:
                        error = (target_state-control_state)[INTEGRATED_COORDINATES]
                        error[1] = (error[1]+np.pi)%(2*np.pi)-np.pi
                        if not previous_saturated:
                            integrated_error = np.clip(integrated_error+CONTROL_PERIOD*error,
                                                       -np.array([.2,.3,.1,.1,.1,.1]), np.array([.2,.3,.1,.1,.1,.1]))
                        multibody_input = design['feedforward']-nominal[446:566].reshape(6,20)@np.r_[control_state-target_state,integrated_error]
                    if actuator_aware:
                        augmented = np.r_[control_state-target_state,
                            observer.motor_states[round(time,6)]-design['feedforward']]
                        multibody_input = design['feedforward']-observer.motor_design['gain']@augmented
                    torque = sum(multibody_input[:2])
                if hybrid:
                    port = speed/WHEEL_RADIUS-gyro[1]
                    energy = nominal[4]*9.81*nominal[3]*(1-np.cos(lean)) + 0.5*nominal[4]*speed**2
                    if case.active:
                        observed_state = control_state
                        velocity = observed_state[7:]
                        port += (sum(velocity[3:5])-sum(velocity[5:7]))/2
                        posture_error = observed_state[3:7]-design["equilibrium"][3:7]
                        energy = nominal[4]*9.81*nominal[3]*(1-np.cos(lean)) + .5*velocity@design["reduced_mass"]@velocity + 10*posture_error@posture_error
                    torque -= (energy_gain*energy+damping_gain)*port
                if integral_gain:
                    # Bounded local integral assistance; no global PH/passivity claim.
                    if previous_estimate_time is not None:
                        integral = float(np.clip(integral+CONTROL_PERIOD*position, -.2/integral_gain, .2/integral_gain))
                    previous_estimate_time = time
                    torque += integral_gain*integral
                heading_target = np.deg2rad(case.heading) if time >= 3 else 0.
                yaw_error = (heading_target-yaw+np.pi)%(2*np.pi)-np.pi
                turn = np.clip(0.7*yaw_error-0.12*gyro[2], -0.5, 0.5)
                commands[:2] = [0.5*torque-turn, 0.5*torque+turn]
                if case.active:
                    commands[:] = multibody_input
                    commands[:2] += .5*(torque-sum(multibody_input[:2]))
                if rotor_damping:
                    # Dissipate unmodelled free-wheel modes using causal encoders.
                    commands[:2] -= rotor_damping*observer.last.joint_velocity[:2]*model.actuator_gear[:2, 0]
                saturated_steps += int(np.any((commands < model.actuator_ctrlrange[:, 0]) | (commands > model.actuator_ctrlrange[:, 1])))
                previous_saturated = bool(np.any((commands < model.actuator_ctrlrange[:, 0]) | (commands > model.actuator_ctrlrange[:, 1])))
                commands = np.clip(commands, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                if case.active:
                    before_slew = commands.copy()
                    commands = np.clip(commands, previous_commands-50*CONTROL_PERIOD, previous_commands+50*CONTROL_PERIOD)
                    previous_saturated |= bool(np.any(abs(commands-before_slew) > 1e-12))
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
            actual_commands.append(applied.copy())
        if reason or step == round(case.duration/TIMESTEP):
            break
        applied = commands.copy() if case.motor_tau == 0 else applied+(1-np.exp(-TIMESTEP/case.motor_tau))*(commands-applied)
        data.ctrl[:] = applied
        before = data.qpos[model.jnt_qposadr[joints]].copy()
        data.xfrc_applied[:] = 0.
        if 15 <= time < 15.2:
            data.xfrc_applied[torso, 4] = case.pulse
        if 35 <= time < 35.2:
            data.xfrc_applied[torso, 4] = -case.pulse
        if 25 <= time < 25.2:
            data.xfrc_applied[torso, 1] = case.lateral
        if time >= case.persistent_start:
            data.xfrc_applied[torso, 4] += case.persistent_torque
        mujoco.mj_step(model, data)
        joint_work = applied*model.actuator_gear[:, 0]*(data.qpos[model.jnt_qposadr[joints]]-before)
        signed_work += joint_work
        positive_work += np.maximum(joint_work, 0.)
        actual_effort += abs(applied)*TIMESTEP
    history = np.array(records)
    qpositions = np.array(qpositions)
    history[:, 4:7] = coordinates(robot, qpositions)
    metrics = measure(history, case, reason)
    metrics.update(mechanical_signed_work_j=signed_work.tolist(), mechanical_positive_work_j=positive_work.tolist(),
                   actual_joint_effort_nm_s=actual_effort.tolist(), actual_wheel_effort_nm_s=float(sum(actual_effort[:2])),
                   actual_all_effort_nm_s=float(sum(actual_effort)),
                   mechanical_total_positive_work_j=float(sum(positive_work)), saturated_control_steps=saturated_steps,
                   dropped_packets=sensor.dropped, stale_packets=sensor.stale, maximum_estimate_age_s=max_estimate_age,
                   revised_observer=revised, integral_gain=integral_gain, lead_s=lead_s, actuator_aware=actuator_aware, rotor_damping=rotor_damping, integral_assisted=integral_assisted,
                   lean_estimate_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((history[:, 8]-history[:, 2])**2)))))
    metrics.update(initial_nonwheel_contacts=initial_contacts, peak_joint_speed=max_speed.tolist())
    robot.sensor_history = sensor.history
    robot.delivery_log = np.array(sensor.delivery_log)
    robot.applied_commands = np.array(actual_commands)
    return robot, history, qpositions, np.array(qvelocities), metrics


def worker(output, name, case_dict, controller, table):
    start = time.perf_counter()
    case = StressCase(**case_dict)
    robot, h, qpos, qvel, metrics = run(case, np.asarray(table),
        hybrid=controller not in ("lqr", "lead_lqr", "actuator_lqr", "rotor_lqr", "soft_lqr", "lqi"), revised=controller != "legacy",
        integral_assisted=controller in ('lqi','lqi_ph'),
        actuator_aware=controller in ('actuator_ph','actuator_lqr','rotor_ph','rotor_lqr'),
        rotor_damping=.01 if controller in ('rotor_ph','rotor_lqr') else 0.,
        lead_s=.015 if controller in ("lead_ph", "lead_lqr") else 0.,
        integral_gain=.5 if controller == "integral_ph" else 0.)
    directory = Path(output)/name
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory/"trajectory.npz", history=h, qpos=qpos, qvel=qvel,
                        applied=robot.applied_commands, delivery=robot.delivery_log)
    observations = np.array([[o.sampled_at, o.available_at, *o.attitude, *o.gyro,
                             *o.joint_position, *o.joint_velocity] for o in robot.sensor_history])
    np.savez_compressed(directory/"observations.npz", observations=observations)
    mujoco.mj_saveLastXML(str(directory/"model.xml"), robot.model)
    metrics.update(episode=name, controller=controller, wall_seconds=time.perf_counter()-start)
    (directory/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def aggregate(rows):
    result = {}
    for group in sorted({r["episode"].rsplit("/", 1)[0] for r in rows}):
        subset = [r for r in rows if r["episode"].rsplit("/", 1)[0] == group]
        successes = [r for r in subset if r["success"]]
        reasons = {}
        for r in subset:
            if not r["success"]:
                reason = r["failure"] or ",".join(r["failed_metric_gates"])
                reasons[reason] = reasons.get(reason, 0)+1
        result[group] = dict(episodes=len(subset), successes=len(successes),
            wilson_95=wilson(len(successes), len(subset)), failures=reasons,
            mean_successful_positive_work_j=None if not successes else float(np.mean([r["mechanical_total_positive_work_j"] for r in successes])),
            mean_successful_wheel_effort_nm_s=None if not successes else float(np.mean([r["wheel_absolute_torque_integral"] for r in successes])))
    return result


def mapping_jobs():
    stresses = [("baseline", {})]
    for field, values in dict(delay=[.02, .04, .08], jitter=[.005, .015, .03],
        dropout=[.1, .3, .5], pitch_bias_deg=[.25, .5, 1.],
        motor_tau=[.01, .03, .06], lateral=[3., 10., 30.],
        friction=[.4, .25, .15], persistent_torque=[.05, .15, .3]).items():
        stresses.extend((f"{field}_{value}", {field: value}) for value in values)
    jobs = []
    for label, changes in stresses:
        for seed in (410000, 410001):
            c = StressCase(seed=seed, pitch=(-1 if seed % 2 else 1)*8., noise_deg=.35, **changes)
            jobs.append((f"{label}/legacy/{seed}", asdict(c), "legacy"))
    return jobs


def execute(output, jobs, workers):
    output = Path(output)
    if not jobs or workers < 1:
        raise ValueError('Nonempty jobs and positive worker count required')
    names = set()
    for name, parameters, controller in jobs:
        if (Path(name).is_absolute() or '..' in Path(name).parts or name in names
                or controller not in ('legacy','hybrid','lqr','lead_ph','lead_lqr','integral_ph','actuator_ph','actuator_lqr','rotor_ph','rotor_lqr','soft_ph','soft_lqr','lqi','lqi_ph')):
            raise ValueError('Invalid or duplicate job name/controller')
        StressCase(**parameters)
        names.add(name)
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = stress_hashes()
    for name in source_hashes:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    manifest = dict(jobs=jobs, source_sha256=source_hashes, columns=COLUMNS,
        versions=dict(python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__),
        scope="Versioned stress test; event-aware recovery includes lateral and persistent pulses. Mechanical work is not electrical energy.",
        assumptions="Fused IMU errors and first-order torque lag are declared simulation assumptions, not calibrated hardware measurements.")
    (output/"protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tables = {active: schedule(active, r=.6) for active in {j[1]["active"] for j in jobs}}
    if any(j[2] in ('actuator_ph','actuator_lqr','rotor_ph','rotor_lqr') for j in jobs):
        tables[True] = actuator_schedule(tables[True])
    soft = soft_schedule(schedule(True, r=.6)) if any(j[2] in ('soft_ph','soft_lqr') for j in jobs) else None
    integral = integral_schedule(schedule(True, r=.6)) if any(j[2] in ('lqi','lqi_ph') for j in jobs) else None
    for active, table in tables.items():
        np.save(output/f"schedule_{active}.npy", table)
    if soft is not None:
        np.save(output/'schedule_soft.npy', soft)
    if integral is not None:
        np.save(output/'schedule_integral.npy', integral)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, str(output), name, c, controller,
                   integral if controller in ('lqi','lqi_ph') else soft if controller in ('soft_ph','soft_lqr') else tables[c["active"]])
                   for name, c, controller in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with (output/"journal.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
            print(f"{len(rows)}/{len(jobs)} {row['episode']} success={row['success']} failure={row['failure']} gates={row['failed_metric_gates']}", flush=True)
    rows.sort(key=lambda r: r["episode"])
    (output/"metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(aggregate(rows), indent=2), encoding="utf-8")
    for name, digest in source_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed during evaluation: '+name)


def stress_hashes():
    result = hashes()
    for name in ('experiments/upkie_stress.py','experiments/audit_upkie_3d_native.py'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def frozen_jobs(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen stress protocol digest changed')
    protocol = json.loads(path.read_text())
    if protocol['source_sha256'] != stress_hashes():
        raise ValueError('Frozen stress source changed')
    return protocol['jobs']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", help="Explicit versioned jobs JSON; omitted for development mapping")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen", help="Hash-checked held-out protocol JSON")
    args = parser.parse_args()
    if args.frozen and args.jobs:
        parser.error('Choose frozen protocol or development jobs')
    jobs = frozen_jobs(args.frozen) if args.frozen else json.loads(Path(args.jobs).read_text()) if args.jobs else mapping_jobs()
    execute(args.output, jobs, args.workers)

