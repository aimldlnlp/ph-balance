"""Local multibody LQR for active sagittal legs, with a delayed state observer.

No constraints are added to the simulator. The no-slip reduction is used only
for nominal local design; physical roll, yaw, slip and contacts stay free.
"""
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from scipy.spatial.transform import Rotation

from src.simulation.upkie import CONTROL_PERIOD, WHEEL_RADIUS


def active_design(robot):
    m, d = robot.model, robot.initial_data()
    wheel_dofs = m.jnt_dofadr[robot.wheel_ids]
    leg_dofs = m.jnt_dofadr[robot.leg_ids]
    projection = np.zeros((m.nv, 7))
    projection[0, 0], projection[4, 1], projection[5, 2] = 1., 1., 1.
    projection[leg_dofs, 3:] = np.eye(4)
    jacobians = []
    for side in ("left", "right"):
        body = m.body(side+"_wheel_center").id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, d.xpos[body], body)
        jacobians.append((jp, jr))
    vertical = np.array([[j[0][2, 2], j[0][2, 3]] for j in jacobians])
    for k in range(1, 7):
        projection[[2, 3], k] = np.linalg.solve(vertical, -np.array([j[0][2]@projection[:, k] for j in jacobians]))
        projection[0, k] = -sum(j[0][0]@projection[:, k] for j in jacobians)/2
    for dof, sign, (jp, jr) in zip(wheel_dofs, (1, -1), jacobians):
        projection[dof] = sign*(jp[0]@projection/WHEEL_RADIUS-jr[1]@projection)
    full = np.empty((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    reduced = projection.T@full@projection
    equilibrium = np.r_[0., robot.trim, 0., d.qpos[m.jnt_qposadr[robot.leg_ids]]]

    p = mujoco.MjData(m)
    def potential(q):
        p.qpos[:] = d.qpos
        p.qpos[3:7] = [1, 0, 0, 0]
        p.qpos[m.jnt_qposadr[robot.leg_ids]] = q[3:7]
        mujoco.mj_forward(m, p)
        centers = np.array([p.xpos[m.body(s+"_wheel_center").id].copy() for s in ("left", "right")])
        delta = centers[0]-centers[1]
        roll = np.arcsin(np.tan(q[1])*delta[0]/np.hypot(delta[1], delta[2]))-np.arctan2(delta[2], delta[1])
        rotation = Rotation.from_euler("ZYX", [q[2], q[1], roll])
        quat = rotation.as_quat()
        p.qpos[3:7] = quat[[3, 0, 1, 2]]
        local = centers.mean(axis=0)-d.qpos[:3]
        p.qpos[:3] = [q[0], 0., WHEEL_RADIUS-.0002]-rotation.apply(local)
        mujoco.mj_forward(m, p)
        return float(9.81*m.body_mass@p.xipos[:, 2])

    h = 1e-4
    unit = np.eye(7)*h
    gradient = np.array([(potential(equilibrium+u)-potential(equilibrium-u))/(2*h) for u in unit])
    stiffness = np.array([[(potential(equilibrium+u+v)-potential(equilibrium+u-v)-potential(equilibrium-u+v)+potential(equilibrium-u-v))/(4*h*h) for v in unit] for u in unit])
    motors = np.zeros((m.nv, 6))
    motors[np.r_[wheel_dofs, leg_dofs], np.arange(6)] = [1, -1, 1, 1, 1, 1]
    input_map = projection.T@motors
    feedforward = np.linalg.lstsq(input_map, gradient, rcond=None)[0]
    if np.max(abs(input_map@feedforward-gradient)) > .02:
        raise ValueError("Nominal posture is not close to static equilibrium")
    a = np.block([[np.zeros((7, 7)), np.eye(7)], [-np.linalg.solve(reduced, stiffness), np.zeros((7, 7))]])
    b = np.vstack([np.zeros((7, 6)), np.linalg.solve(reduced, input_map)])
    ad, bd, *_ = cont2discrete((a, b, np.eye(14), np.zeros((14, 6))), CONTROL_PERIOD)
    q = np.diag([80., 150., 20., 100., 100., 100., 100., 15., 8., 1., 1., 1., 1., 1.])
    r = np.diag([.3, .3, .04, .04, .04, .04])
    riccati = solve_discrete_are(ad, bd, q, r)
    gain = np.linalg.solve(r+bd.T@riccati@bd, bd.T@riccati@ad)
    poles = np.linalg.eigvals(ad-bd@gain)
    if max(abs(poles)) >= 1:
        raise ValueError("Active local LQR is unstable")
    return dict(gain=gain, ad=ad, bd=bd, equilibrium=np.r_[equilibrium, np.zeros(7)],
                feedforward=feedforward, reduced_mass=reduced, pole_magnitudes=abs(poles))


class ActiveObserver:
    """Fixed-lag nominal prediction using only time-stamped sensor packets."""
    def __init__(self, design):
        self.design = design
        self.state = design["equilibrium"].copy()
        self.last = None
        self.commands = {}
        self.yaw = 0.
        self.xy = np.zeros(2)
        self.initial_encoder = None

    def command(self, time, torque):
        self.commands[round(time, 6)] = np.array(torque, copy=True)

    def predict(self, state, time):
        d = self.design
        return d["equilibrium"]+d["ad"]@(state-d["equilibrium"])+d["bd"]@(
            self.commands.get(round(time, 6), d["feedforward"])-d["feedforward"])

    def update(self, observation, now):
        if observation.available_at > now+1e-12 or (self.last is not None and observation.sampled_at <= self.last.sampled_at):
            raise ValueError("Unavailable or out-of-order observation")
        p, v = observation.joint_position, observation.joint_velocity
        pitch, yaw = observation.attitude[1:3]
        rate, yaw_rate = observation.gyro[1:3]
        encoder = WHEEL_RADIUS*((p[0]-p[1])/2+pitch+(-sum(p[2:4])+sum(p[4:6]))/2)
        if self.last is None:
            self.initial_encoder = encoder
            self.state[1:7] = np.r_[pitch, yaw, p[2:]]
        else:
            self.state = self.predict(self.state, self.last.sampled_at)
        speed = WHEEL_RADIUS*((v[0]-v[1])/2+rate+(-sum(v[2:4])+sum(v[4:6]))/2)
        measurement = np.r_[encoder-self.initial_encoder, pitch, yaw, p[2:], speed, rate, yaw_rate, v[2:]]
        innovation = measurement-self.state
        innovation[2] = (innovation[2]+np.pi)%(2*np.pi)-np.pi
        self.state += np.array([.05, .04, .04, .9, .9, .9, .9, .02, .7, .7, .7, .7, .7, .7])*innovation
        self.last = observation

    def estimate(self, now):
        if self.last is None:
            return None
        age = now-self.last.sampled_at
        if age < -1e-12:
            raise ValueError("Estimator queried in the past")
        state = self.state.copy()
        for k in range(round(age/CONTROL_PERIOD)):
            state = self.predict(state, self.last.sampled_at+k*CONTROL_PERIOD)
        attitude = np.array([self.last.attitude[0], state[1], state[2]])
        gyro = np.array([self.last.gyro[0], state[8], state[9]])
        return attitude, gyro, np.array([state[0], 0.]), state[7]

    def active_state(self, now):
        state = self.state.copy()
        for k in range(round((now-self.last.sampled_at)/CONTROL_PERIOD)):
            state = self.predict(state, self.last.sampled_at+k*CONTROL_PERIOD)
        return state
