"""Planar, mechanically locked-leg Upkie contact model and local LQR design.

Source masses, inertias, meshes, and wheel limits come from the vendored URDF.
The planar constraints and mechanical leg locks are experiment assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from src.controllers.port_hamiltonian import HybridPortHamiltonianController
from src.models.parameters import RobotParameters
from src.models.state import RobotState

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "upkie"
SOURCE_REVISION = "94735fbe6137276a41de0ff4cc04d2e533fa9e33"
LEG_POSE = {"left_hip": -0.35, "left_knee": 0.50, "right_hip": 0.35, "right_knee": -0.50}
TIMESTEP = 0.001
CONTROL_PERIOD = 0.005
WHEEL_RADIUS = 0.05
WHEEL_TORQUE_LIMIT = 1.7
WHEEL_SPEED_LIMIT = 111.0


@dataclass
class UpkieModel:
    model: mujoco.MjModel
    parameters: RobotParameters
    trim: float
    motor_signs: np.ndarray
    wheel_joint_ids: np.ndarray
    axle_height: float

    def initial_data(self, angle_degrees: float = 0.0) -> mujoco.MjData:
        if not np.isfinite(angle_degrees):
            raise ValueError("Initial angle must be finite")
        data = mujoco.MjData(self.model)
        data.qpos[1] = self.axle_height
        data.qpos[2] = self.trim + np.deg2rad(angle_degrees)
        mujoco.mj_forward(self.model, data)
        return data

    def state(self, data: mujoco.MjData) -> RobotState:
        return RobotState(float(data.qpos[2] - self.trim), float(data.qvel[2]), float(data.qpos[0]), float(data.qvel[0]))

    def body_ground_contact(self, data: mujoco.MjData) -> bool:
        """Distinguish an ankle/body strike from normal wheel-ground contact."""
        chassis = self.model.body("chassis").id
        return any(chassis in self.model.geom_bodyid[contact.geom] for contact in data.contact)


def build_upkie(mass_scale: float = 1.0, *, width: int = 1280, height: int = 720) -> UpkieModel:
    """Import the original URDF, freeze four leg joints, and add planar motion."""
    if not np.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError("mass_scale must be positive and finite")
    if width <= 0 or height <= 0:
        raise ValueError("Render dimensions must be positive")
    urdf = ET.parse(ASSETS / "urdf" / "upkie.urdf").getroot()
    for mesh in urdf.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("package://upkie_description/", ASSETS.as_posix() + "/"))
    # Virtual frame visuals have zero volume and cannot be compiled as geoms.
    for link in urdf.findall("link"):
        for visual in link.findall("visual"):
            box = visual.find("geometry/box")
            if box is not None and box.get("size") == "0 0 0":
                link.remove(visual)
    ET.SubElement(ET.SubElement(urdf, "mujoco"), "compiler", discardvisual="false", fusestatic="false", balanceinertia="true")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    with TemporaryDirectory() as temporary:
        converted = Path(temporary) / "upkie.xml"
        mujoco.mj_saveLastXML(str(converted), imported)
        root = ET.parse(converted).getroot()
    root.find("compiler").set("fusestatic", "false")
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name in LEG_POSE:
                quaternion = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
                rotation = np.empty(4)
                result = np.empty(4)
                mujoco.mju_axisAngle2Quat(rotation, np.fromstring(joint.get("axis"), sep=" "), LEG_POSE[name])
                mujoco.mju_mulQuat(result, quaternion, rotation)
                body.set("quat", " ".join(map(str, result)))
                body.remove(joint)
    locked = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    pose = mujoco.MjData(locked)
    mujoco.mj_forward(locked, pose)
    # These named frames are preserved until the final fixed-body fusion.
    centers = np.array([pose.xpos[locked.body(f"{side}_wheel_center").id] for side in ("left", "right")])
    if not np.allclose(centers[0, [0, 2]], centers[1, [0, 2]], atol=1e-6):
        raise ValueError("Locked legs must produce coaxial wheels")
    axle = centers.mean(axis=0)
    root.find("compiler").set("fusestatic", "true")
    world = root.find("worldbody")
    original_base = world.find("body")
    world.remove(original_base)
    original_base.set("pos", " ".join(map(str, -axle)))
    chassis = ET.SubElement(world, "body", name="chassis")
    ET.SubElement(chassis, "joint", name="position", type="slide", axis="1 0 0", limited="false")
    ET.SubElement(chassis, "joint", name="height", type="slide", axis="0 0 1", limited="false")
    ET.SubElement(chassis, "joint", name="pitch", type="hinge", axis="0 1 0", limited="false")
    chassis.append(original_base)
    for inertial in root.iter("inertial"):
        inertial.set("mass", str(float(inertial.get("mass")) * mass_scale))
        inertia = np.fromstring(inertial.get("diaginertia"), sep=" ") * mass_scale
        # URDF virtual frames use point masses; MuJoCo fusion needs positive inertia.
        inertia = np.maximum(inertia, 1e-10 * mass_scale)
        inertial.set("diaginertia", " ".join(map(str, inertia)))
    for geom in root.iter("geom"):
        if geom.get("group") == "1":
            geom.set("rgba", geom.get("rgba", "0.5 0.5 0.5 1").rsplit(" ", 1)[0] + " 1")
        else:
            geom.set("contype", "1")
            geom.set("conaffinity", "0")
            geom.set("friction", "1.0 0.005 0.0001")
            geom.set("condim", "3")
            geom.set("solref", "0.01 1")
    ET.SubElement(root, "option", timestep=str(TIMESTEP), gravity="0 0 -9.81", integrator="implicitfast", iterations="80")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1", rgba="0.86 0.86 0.83 1", contype="0", conaffinity="1", friction="1.0 0.005 0.0001", condim="3", solref="0.01 1")
    ET.SubElement(world, "light", pos="-1 -2 4", dir="0.2 0.4 -1", directional="true", diffuse="0.45 0.45 0.45")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.4 0.4 0.4", specular="0 0 0")
    ET.SubElement(root.find("asset"), "texture", name="sky", type="skybox", builtin="flat", rgb1="0.98 0.973 0.95", rgb2="0.98 0.973 0.95", width="32", height="32")
    actuators = ET.SubElement(root, "actuator")
    # The URDF configures the left and right motors in opposite directions.
    for side, sign in (("left", 1), ("right", -1)):
        ET.SubElement(actuators, "motor", name=f"{side}_motor", joint=f"{side}_wheel", gear=str(sign), ctrllimited="true", ctrlrange=f"{-WHEEL_TORQUE_LIMIT} {WHEEL_TORQUE_LIMIT}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("chassis").id
    wheel_bodies = [model.jnt_bodyid[model.joint(f"{side}_wheel").id] for side in ("left", "right")]
    mass = model.body_mass[body]
    com = data.xipos[body]
    trim = -np.arctan2(com[0], com[2])
    length = float(np.hypot(com[0], com[2]))
    rotation = data.ximat[body].reshape(3, 3)
    inertia = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
    wheel_inertias = []
    for wheel in wheel_bodies:
        rotation = data.ximat[wheel].reshape(3, 3)
        wheel_inertias.append((rotation @ np.diag(model.body_inertia[wheel]) @ rotation.T)[1, 1])
    parameters = RobotParameters(float(mass), float(np.mean(model.body_mass[wheel_bodies])), float(inertia[1, 1]), float(np.mean(wheel_inertias)), length, WHEEL_RADIUS, 9.81, 0.0, 0.0)
    joint_ids = np.array([model.joint(f"{side}_wheel").id for side in ("left", "right")])
    signs = np.array([1.0, -1.0])
    # Contact compression offsets the gravity preload without a settling kick.
    return UpkieModel(model, parameters, float(trim), signs, joint_ids, WHEEL_RADIUS - 0.0002)


def lqr_design(robot: UpkieModel, q_weights: tuple = (150.0, 8.0, 80.0, 15.0), r_weight: float = 0.15) -> tuple[np.ndarray, dict]:
    """Design sampled LQR from the source inertias under local no-slip rolling."""
    if len(q_weights) != 4 or not np.isfinite(q_weights).all() or min(q_weights) <= 0 or not np.isfinite(r_weight) or r_weight <= 0:
        raise ValueError("LQR weights must be finite and positive")
    model = robot.model
    data = robot.initial_data()
    full_mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_mass)
    projection = np.zeros((model.nv, 2))
    projection[0, 0], projection[2, 1] = 1, 1
    for joint, sign in zip(robot.wheel_joint_ids, robot.motor_signs):
        projection[model.jnt_dofadr[joint]] = (sign / WHEEL_RADIUS, -sign)
    reduced_mass = projection.T @ full_mass @ projection
    p = robot.parameters
    acceleration = np.linalg.solve(reduced_mass, np.array([0.0, p.body_mass * p.gravity * p.center_of_mass_length]))
    input_acceleration = np.linalg.solve(reduced_mass, np.array([1 / WHEEL_RADIUS, -1.0]))
    a = np.zeros((4, 4))
    a[0, 1], a[2, 3] = 1, 1
    a[1, 0], a[3, 0] = acceleration[1], acceleration[0]
    b = np.array([[0], [input_acceleration[1]], [0], [input_acceleration[0]]])
    ad, bd, _, _, _ = cont2discrete((a, b, np.eye(4), np.zeros((4, 1))), CONTROL_PERIOD)
    q, r = np.diag(q_weights), np.array([[r_weight]])
    riccati = solve_discrete_are(ad, bd, q, r)
    gains = np.linalg.solve(r + bd.T @ riccati @ bd, bd.T @ riccati @ ad).ravel()
    poles = np.linalg.eigvals(ad - bd @ gains.reshape(1, 4))
    if not np.all(np.abs(poles) < 1):
        raise ValueError("LQR linearization is not locally stable")
    return gains, {"A": a.tolist(), "B": b.tolist(), "reduced_mass_matrix": reduced_mass.tolist(), "Q": list(q_weights), "R": r_weight, "gains": gains.tolist(), "closed_loop_pole_magnitudes": np.abs(poles).tolist()}


def hybrid_controller(gains: np.ndarray, energy_gain: float = 0.1, damping_gain: float = 0.01) -> HybridPortHamiltonianController:
    return HybridPortHamiltonianController(energy_gain, damping_gain, tuple(float(gain) for gain in gains), torque_limit=2 * WHEEL_TORQUE_LIMIT)


__all__ = ["UpkieModel", "build_upkie", "lqr_design", "hybrid_controller"]
