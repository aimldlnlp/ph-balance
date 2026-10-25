"""Measure initial nonwheel clearance from the retained source primitives.

This only evaluates geometry; it does not lift, clamp or otherwise alter a run.
The two wheel contacts deliberately start with the documented 0.2 mm preload.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from src.simulation.upkie_3d import build_floating


def floor_clearances(robot, data):
    model = robot.model
    distances = {}
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        if name == "floor" or not model.geom_contype[geom] or "wheel" in name:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        size = model.geom_size[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            extent = abs(rotation[2])@size
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axial = rotation[2, 2]
            extent = size[1]*abs(axial)+size[0]*np.sqrt(max(0., 1-axial*axial))
        else:
            raise ValueError(f"Unsupported source collision primitive: {name}")
        distances[name] = float(data.geom_xpos[geom, 2]-extent)
    return distances


def inspect():
    rows = []
    for active in (False, True):
        for knee in ([.5, .8, 1.2] if active else [.5]):
            for scale in (.8, 1., 1.2):
                robot = build_floating(active=active, knee=knee, torso_scale=scale)
                for pitch in (-10., 0., 10.):
                    for heading in (-15., 15.):
                        data = robot.initial_data(pitch, heading)
                        distances = floor_clearances(robot, data)
                        minimum = min(distances, key=distances.get)
                        contacts = robot.nonwheel_ground_contacts(data)
                        rows.append(dict(active=active, knee_rad=knee, torso_scale=scale, pitch_deg=pitch,
                                         heading_deg=heading, root_position=data.qpos[:3].tolist(),
                                         minimum_nonwheel_clearance_m=distances[minimum], closest_geom=minimum,
                                         nonwheel_ground_contacts=contacts, all_nonwheel_clearances_m=distances))
    release = [r for r in rows if r["knee_rad"] == .5]
    return dict(cases=rows, expanded_envelope_valid=all(not r["nonwheel_ground_contacts"] and r["minimum_nonwheel_clearance_m"] > 0 for r in rows),
                protocol_release_valid=all(not r["nonwheel_ground_contacts"] and r["minimum_nonwheel_clearance_m"] > 0 for r in release),
                protocol_release_cases=len(release), protocol_release_knee_rad=.5,
                minimum_protocol_release_clearance_m=min(r["minimum_nonwheel_clearance_m"] for r in release),
                definition="Exact box/cylinder support in world Z above the z=0 floor. Visual meshes are not collision geometry. No geometry/state is changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result = inspect()
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(protocol_release_valid=result["protocol_release_valid"], expanded_envelope_valid=result["expanded_envelope_valid"], cases=len(result["cases"]),
                          minimum_clearance_m=min(r["minimum_nonwheel_clearance_m"] for r in result["cases"]))))
