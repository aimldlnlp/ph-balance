"""Source-backed floating Upkie, with no base constraints or ground truth control.

The source URDF is unchanged. Mechanical leg locks are baked into the geometry;
active legs retain the source joints and use bounded supporting posture torques.
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
from scipy.spatial.transform import Rotation

from src.simulation.upkie import ASSETS, TIMESTEP, CONTROL_PERIOD, WHEEL_RADIUS


def leg_pose(knee: float = 0.5) -> dict[str, float]:
    return {"left_hip": -0.1-knee/2, "left_knee": knee,
            "right_hip": 0.1+knee/2, "right_knee": -knee}


@dataclass
class FloatingUpkie:
    model: mujoco.MjModel
    axle: np.ndarray
    trim: float
    length: float
    body_mass: float
    wheel_ids: np.ndarray
    leg_ids: np.ndarray
    limits: dict
    knee: float

    def initial_data(self, pitch_deg=0.0, yaw_deg=0.0):
        if not np.isfinite([pitch_deg, yaw_deg]).all() or abs(pitch_deg) >= 85:
            raise ValueError("Invalid initial orientation")
        data = mujoco.MjData(self.model)
        rotation = Rotation.from_euler("ZY", [np.deg2rad(yaw_deg), self.trim+np.deg2rad(pitch_deg)])
        quat = rotation.as_quat()
        data.qpos[3:7] = quat[[3, 0, 1, 2]]
        data.qpos[:3] = np.array([0., 0., WHEEL_RADIUS-0.0002]) - rotation.apply(self.axle)
        for j in self.leg_ids:
            data.qpos[self.model.jnt_qposadr[j]] = leg_pose(self.knee)[self.model.joint(j).name]
        mujoco.mj_forward(self.model, data)
        return data

    def nonwheel_ground_contacts(self, data):
        floor = self.model.geom("floor").id
        return [self.model.geom(int(pair[1] if pair[0] == floor else pair[0])).name
                for contact in data.contact for pair in [contact.geom]
                if floor in pair and "wheel" not in (self.model.geom(int(pair[1] if pair[0] == floor else pair[0])).name or "")]

    def orientation(self, data):
        return Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_euler("xyz")


def build_floating(*, active=False, knee=0.5, torso_scale=1.0, friction=0.8):
    if not np.isfinite([knee, torso_scale, friction]).all() or not 0 < knee < 2.5 or torso_scale <= 0 or friction <= 0:
        raise ValueError("Invalid model parameters")
    source = ET.parse(ASSETS / "urdf/upkie.urdf").getroot()
    limits = {j.get("name"): {k: float(v) for k, v in j.find("limit").attrib.items()}
              for j in source.findall("joint") if j.find("limit") is not None}
    torso = source.find("link[@name='torso']/inertial")
    torso.find("mass").set("value", str(float(torso.find("mass").get("value"))*torso_scale))
    for key, value in torso.find("inertia").attrib.items():
        torso.find("inertia").set(key, str(float(value)*torso_scale))
    for mesh in source.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("package://upkie_description/", ASSETS.as_posix()+"/"))
    for link in source.findall("link"):
        for visual in list(link.findall("visual")):
            box = visual.find("geometry/box")
            if box is not None and box.get("size") == "0 0 0":
                link.remove(visual)
    ET.SubElement(ET.SubElement(source, "mujoco"), "compiler", discardvisual="false", fusestatic="false", balanceinertia="true")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
    with TemporaryDirectory() as directory:
        path = Path(directory)/"model.xml"
        mujoco.mj_saveLastXML(str(path), imported)
        root = ET.parse(path).getroot()
    root.find("compiler").set("fusestatic", "false")
    pose = leg_pose(knee)
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name in pose and not active:
                rotation = np.empty(4)
                result = np.empty(4)
                mujoco.mju_axisAngle2Quat(rotation, np.fromstring(joint.get("axis"), sep=" "), pose[name])
                mujoco.mju_mulQuat(result, np.fromstring(body.get("quat", "1 0 0 0"), sep=" "), rotation)
                body.set("quat", " ".join(map(str, result)))
                body.remove(joint)
    for inertial in root.iter("inertial"):
        inertial.set("diaginertia", " ".join(map(str, np.maximum(np.fromstring(inertial.get("diaginertia"), sep=" "), 1e-10))))
    world = root.find("worldbody")
    base = world.find("body")
    ET.SubElement(base, "freejoint", name="floating_base")
    for i, geom in enumerate(root.iter("geom")):
        # Collision names retain source body identity for failure classification.
        parent = next(b for b in root.iter("body") if geom in list(b))
        geom.set("name", f"{parent.get('name')}_{i}")
        if geom.get("group") != "1":
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("friction", f"{friction} 0.005 0.0001")
            geom.set("condim", "3")
            geom.set("solref", "0.01 1")
        else:
            geom.set("rgba", geom.get("rgba", "0.5 0.5 0.5 1").rsplit(" ", 1)[0]+" 1")
    ET.SubElement(root, "option", timestep=str(TIMESTEP), gravity="0 0 -9.81", integrator="implicitfast", iterations="80")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1", rgba="0.90 0.90 0.88 1", contype="0", conaffinity="1", friction=f"{friction} 0.005 0.0001", condim="3", solref="0.01 1")
    ET.SubElement(world, "light", pos="-1 -2 4", dir="0.2 0.4 -1", directional="true")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5")
    actuators = ET.SubElement(root, "actuator")
    for name in ["left_wheel", "right_wheel"] + (list(pose) if active else []):
        limit = limits[name]["effort"]
        sign = -1 if name == "right_wheel" else 1
        ET.SubElement(actuators, "motor", name=name+"_motor", joint=name, gear=str(sign), ctrllimited="true", ctrlrange=f"{-limit} {limit}", forcelimited="true", forcerange=f"{-limit} {limit}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    wheel_ids = np.array([model.joint(s+"_wheel").id for s in ("left", "right")])
    leg_ids = np.array([model.joint(n).id for n in pose], dtype=int) if active else np.array([], dtype=int)
    data = mujoco.MjData(model)
    for j in leg_ids:
        data.qpos[model.jnt_qposadr[j]] = pose[model.joint(j).name]
    mujoco.mj_forward(model, data)
    axle = np.mean([data.xpos[model.body(s+"_wheel_center").id] for s in ("left", "right")], axis=0)
    wheel_bodies = {b for b in range(model.nbody) if "wheel" in (model.body(b).name or "")}
    bodies = [b for b in range(1, model.nbody) if b not in wheel_bodies]
    mass = float(sum(model.body_mass[bodies]))
    com = np.sum(model.body_mass[bodies, None]*data.xipos[bodies], axis=0)/mass-axle
    return FloatingUpkie(model, axle, float(-np.arctan2(com[0], com[2])), float(np.hypot(com[0], com[2])), mass, wheel_ids, leg_ids, limits, knee)


def floating_lqr(robot, q=(150., 8., 80., 15.), r=0.15, *, matrices=False):
    """Nominal no-slip sagittal reduction of the full floating mass matrix."""
    if len(q) != 4 or not np.isfinite([*q, r]).all() or min(q) <= 0 or r <= 0:
        raise ValueError("LQR weights must be finite and positive")
    model, data = robot.model, robot.initial_data()
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    projection = np.zeros((model.nv, 2))
    projection[0, 0], projection[4, 1] = 1, 1
    # Rotation about the axle also translates the floating base origin.
    projection[0, 1] = -robot.axle[2]
    projection[2, 1] = robot.axle[0]
    for joint, sign in zip(robot.wheel_ids, (1., -1.)):
        projection[model.jnt_dofadr[joint]] = (sign/WHEEL_RADIUS, -sign)
    reduced = projection.T@mass@projection
    gravity = np.linalg.solve(reduced, [0., robot.body_mass*9.81*robot.length])
    inp = np.linalg.solve(reduced, [1/WHEEL_RADIUS, -1.])
    a = np.zeros((4, 4))
    a[0, 1], a[2, 3] = 1, 1
    a[1, 0], a[3, 0] = gravity[1], gravity[0]
    b = np.array([[0], [inp[1]], [0], [inp[0]]])
    ad, bd, *_ = cont2discrete((a, b, np.eye(4), np.zeros((4, 1))), CONTROL_PERIOD)
    p = solve_discrete_are(ad, bd, np.diag(q), [[r]])
    gain = np.linalg.solve([[r]]+bd.T@p@bd, bd.T@p@ad).ravel()
    assert np.max(abs(np.linalg.eigvals(ad-bd@gain[None, :]))) < 1
    return (gain, ad, bd.ravel()) if matrices else gain
