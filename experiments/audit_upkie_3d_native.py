"""Re-evaluate world coordinates at saved qpos, without advancing the plant.

MuJoCo mj_step world-frame caches precede the integrated qpos by one physics
step. This postprocessing makes acceptance use the exact saved 200 Hz pose.
The original run, observations, commands and protocol remain untouched.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path

from experiments.evaluate_upkie_3d import hashes, summary
from experiments.upkie_3d_pilot import Case, measure
from src.simulation.upkie_3d import build_floating
import mujoco
import numpy as np


def coordinates(robot, qpos):
    data = mujoco.MjData(robot.model)
    left, right = [robot.model.body(side+"_wheel_center").id for side in ("left", "right")]
    torso = robot.model.body("torso").id
    result = np.empty((len(qpos), 3))
    for i, position in enumerate(qpos):
        data.qpos[:] = position
        mujoco.mj_kinematics(robot.model, data)
        result[i] = ((data.xpos[left, 0]+data.xpos[right, 0])/2,
                     (data.xpos[left, 1]+data.xpos[right, 1])/2, data.xpos[torso, 2])
    return result


def worker(evaluation, output, row):
    source = Path(evaluation)/row["episode"]/"trajectory.npz"
    with np.load(source) as saved:
        h, qpos = saved["history"].copy(), saved["qpos"]
    c = Case(**row["case"])
    robot = build_floating(active=c.active, torso_scale=c.torso_scale, friction=c.friction)
    xyz = coordinates(robot, qpos)
    difference = xyz-h[:, 4:7]
    h[:, 4:7] = xyz
    result = measure(h, c, row["failure"])
    result.update(episode=row["episode"], hybrid=row["hybrid"],
                  original_success=row["success"], decision_unchanged=bool(result["success"] == row["success"]),
                  maximum_cached_xy_difference_m=float(np.max(np.linalg.norm(difference[:, :2], axis=1))),
                  maximum_cached_height_difference_m=float(np.max(abs(difference[:, 2]))),
                  source_native_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    target = Path(output)/row["episode"]
    target.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(target/"coordinates.npz", time=h[:, 0], xy_height=xyz)
    (target/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def audit(evaluation, output, workers):
    evaluation, output = Path(evaluation), Path(output)
    protocol = json.loads((evaluation/"protocol.json").read_text())
    if protocol["source_sha256"] != hashes():
        raise ValueError("Frozen simulation source changed")
    rows = json.loads((evaluation/"metrics.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, str(evaluation), str(output), row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%50 == 0:
                print(f"Native pose audit {len(results)}/{len(rows)}", flush=True)
    results.sort(key=lambda r: r["episode"])
    report = dict(episodes=len(results), all_decisions_unchanged=all(r["decision_unchanged"] for r in results),
                  source_sha256=protocol["source_sha256"], summary=summary(results),
                  maximum_cached_xy_difference_m=max(r["maximum_cached_xy_difference_m"] for r in results),
                  maximum_cached_height_difference_m=max(r["maximum_cached_height_difference_m"] for r in results),
                  methodology="mj_kinematics at every saved qpos, no simulation step, identical frozen acceptance function and transient windows. Original logs retained.")
    (output/"metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output/"verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_decisions_unchanged"]:
        raise ValueError("Native-pose audit changes a success decision; investigate before delivery")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit(args.evaluation, args.output, args.workers)
