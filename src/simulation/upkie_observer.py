"""Timestamped IMU/encoder observations and a causal complementary observer.

Attitude is a simulated fused IMU output, not a raw accelerometer measurement.
No absolute translation/velocity is observed: position uses wheel odometry.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.simulation.upkie import WHEEL_RADIUS


@dataclass(frozen=True)
class Observation:
    sampled_at: float
    available_at: float
    attitude: np.ndarray
    gyro: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


class Sensors:
    def __init__(self, robot, seed, noise_deg=0.2, delay=0.01):
        if not np.isfinite([noise_deg, delay]).all() or min(noise_deg, delay) < 0:
            raise ValueError("Noise and delay must be finite and nonnegative")
        self.robot, self.delay = robot, delay
        self.rng = np.random.default_rng(seed)
        self.noise = np.deg2rad(noise_deg)
        self.pending = deque()
        self.history = []
        self.joints = np.r_[robot.wheel_ids, robot.leg_ids]

    def sample(self, data, time):
        model = self.robot.model
        observation = Observation(float(time), float(time+self.delay),
            self.robot.orientation(data)+self.rng.normal(0, self.noise, 3),
            data.qvel[3:6].copy()+self.rng.normal(0, np.deg2rad(0.05), 3),
            data.qpos[model.jnt_qposadr[self.joints]].copy()+self.rng.normal(0, 1e-4, len(self.joints)),
            data.qvel[model.jnt_dofadr[self.joints]].copy()+self.rng.normal(0, 0.002, len(self.joints)))
        self.pending.append(observation)
        self.history.append(observation)

    def receive(self, time):
        received = []
        while self.pending and self.pending[0].available_at <= time+1e-12:
            received.append(self.pending.popleft())
        return received


class Observer:
    def __init__(self, ad=None, bd=None, trim=0.):
        self.last = None
        self.attitude = np.zeros(3)
        self.xy = np.zeros(2)
        self.speed = 0.
        self.ad, self.bd, self.trim = ad, bd, trim
        self.sagittal = np.zeros(4)
        self.commands = {}
        self.last_encoder = None

    def command(self, time, torque):
        self.commands[round(time, 6)] = float(torque)

    def update(self, observation, now):
        if observation.available_at > now+1e-12 or observation.sampled_at > now:
            raise ValueError("Future observation")
        if self.last is not None and observation.sampled_at <= self.last.sampled_at:
            raise ValueError("Out-of-order observation")
        dt = 0. if self.last is None else observation.sampled_at-self.last.sampled_at
        if self.last is None:
            self.attitude = observation.attitude.copy()
        else:
            predicted = self.attitude + dt*observation.gyro
            innovation = (observation.attitude-predicted+np.pi)%(2*np.pi)-np.pi
            self.attitude = predicted + 0.04*innovation
        # Wheel rotation is relative to the ankle; active hip/knee rates matter.
        rates = observation.joint_velocity[:2]*[1., -1.]+observation.gyro[1]
        if len(observation.joint_velocity) == 6:
            rates += [-sum(observation.joint_velocity[2:4]), sum(observation.joint_velocity[4:6])]
        new_speed = WHEEL_RADIUS*float(np.mean(rates))
        if self.ad is not None:
            if self.last is None:
                self.sagittal[0] = observation.attitude[1]-self.trim
            else:
                self.sagittal = self.ad@self.sagittal+self.bd*self.commands.get(round(self.last.sampled_at, 6), 0.)
            self.sagittal[[0, 1, 3]] += np.array([0.04, 0.7, 0.02])*(
                np.array([observation.attitude[1]-self.trim, observation.gyro[1], new_speed])-self.sagittal[[0, 1, 3]])
            new_speed = self.sagittal[3]
        yaw = self.attitude[2]
        encoder = WHEEL_RADIUS*(float(np.mean(observation.joint_position[:2]*[1., -1.]))+self.attitude[1])
        if len(observation.joint_position) == 6:
            encoder += WHEEL_RADIUS*(-sum(observation.joint_position[2:4])+sum(observation.joint_position[4:6]))/2
        # Position must be anchored to encoders, not integrated nominal-model
        # velocity: mass mismatch otherwise produces unobservable model drift.
        if self.last_encoder is not None:
            self.xy += (encoder-self.last_encoder)*np.array([np.cos(yaw), np.sin(yaw)])
        self.last_encoder = encoder
        self.speed = new_speed if self.ad is not None else self.speed+0.2*(new_speed-self.speed)
        self.last = observation

    def estimate(self, now):
        if self.last is None:
            return None
        age = now-self.last.sampled_at
        if age < -1e-12:
            raise ValueError("Estimator queried in the past")
        attitude = self.attitude + age*self.last.gyro
        xy = self.xy + age*self.speed*np.array([np.cos(attitude[2]), np.sin(attitude[2])])
        gyro, speed = self.last.gyro.copy(), self.speed
        if self.ad is not None:
            predicted = self.sagittal.copy()
            for k in range(round(age/0.005)):
                predicted = self.ad@predicted+self.bd*self.commands.get(round(self.last.sampled_at+k*0.005, 6), 0.)
            attitude[1], gyro[1], speed = predicted[0]+self.trim, predicted[1], predicted[3]
        return attitude, gyro, xy, speed
