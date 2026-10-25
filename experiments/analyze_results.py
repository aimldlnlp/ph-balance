"""Failure-aware statistical checks for recorded evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from scipy.spatial.transform import Rotation

def audit(directory, rows, summary):
    protocol = json.loads((directory/"protocol.json").read_text())
    expected = {(group, c["seed"]) for group, cases in (("hybrid", protocol["robust_cases"]), ("lqr", protocol["robust_cases"]),
                ("combined", protocol["combined_cases"]), ("locked_validation", protocol["locked_validation"])) for c in cases}
    actual = {(row["episode"].split("/")[0], row["case"]["seed"]) for row in rows}
    if len(actual) != len(rows) or actual != expected:
        raise ValueError("Missing, duplicate or unexpected final episodes")
    expected_cases = {(group, case["seed"]): case for group, cases in (("hybrid", protocol["robust_cases"]), ("lqr", protocol["robust_cases"]),
                      ("combined", protocol["combined_cases"]), ("locked_validation", protocol["locked_validation"])) for case in cases}
    checks = []
    paired_noise, paired_noise_checks = {}, []
    for row in rows:
        key = (row["episode"].split("/")[0], row["case"]["seed"])
        if row["case"] != expected_cases[key]:
            raise ValueError("Episode conditions differ from the frozen protocol")
        p = directory/row["episode"]
        native = np.load(p/"trajectory.npz")
        h = native["history"]
        if not len(h) == len(native["qpos"]) == len(native["qvel"]):
            raise ValueError("Misaligned native trajectory")
        displacement = np.linalg.norm(h[:, 4:6]-h[0, 4:6], axis=1)
        if not np.isclose(displacement[-1], row["final_position_m"]) or not np.isclose(max(displacement), row["max_excursion_m"]):
            raise ValueError("Position gate does not match the native evaluation trace")
        completed = row["failure"] is None and np.isclose(h[-1, 0], row["case"]["duration"])
        if completed != row["completed"]:
            raise ValueError("Completion flag contradicts recorded exposure")
        if row["success"] and (not completed or displacement[-1] > .15 or max(displacement) > .5 or any(s is None or s > 3 for s in row["settling_s"])):
            raise ValueError("Successful label contradicts an acceptance gate")
        # Independently recompute the piecewise-constant torque exposure.
        effort = sum((h[i+1, 0]-h[i, 0])*(abs(h[i, 12])+abs(h[i, 13])) for i in range(len(h)-1))
        if not np.isclose(effort, row["wheel_absolute_torque_integral"], rtol=1e-10):
            raise ValueError("Effort calculation mismatch")
        if np.max(abs(h[:, 12:14])) > 1.7+1e-10 or np.max(abs(h[:, 14:18])) > 16+1e-10:
            raise ValueError("Recorded torque outside source limits")
        available = h[:, 18] >= 0
        if np.any(h[available, 18] > h[available, 0]-.01+1e-9):
            raise ValueError("Control used an unavailable observation")
        obs = np.load(p/"observations.npz")["observations"]
        if not np.allclose(obs[:, 1]-obs[:, 0], .01, rtol=0, atol=1e-10):
            raise ValueError("Observation delay mismatch")
        observed_indices = np.searchsorted(h[:, 0], obs[:, 0])
        if not np.allclose(h[observed_indices, 0], obs[:, 0], rtol=0, atol=1e-10):
            raise ValueError("Raw observations do not align with sampled native states")
        actual_attitude = Rotation.from_quat(native["qpos"][observed_indices][:, [4, 5, 6, 3]]).as_euler("xyz")
        noise = (obs[:, 2:5]-actual_attitude+np.pi)%(2*np.pi)-np.pi
        if key[0] in ("hybrid", "lqr"):
            seed = key[1]
            if seed in paired_noise:
                reference = paired_noise.pop(seed)
                count = min(len(reference), len(noise))
                if not np.allclose(reference[:count], noise[:count], atol=1e-13, rtol=0):
                    raise ValueError("Paired controllers received different attitude-noise streams")
                paired_noise_checks.append(dict(seed=seed, common_samples=count,
                                               maximum_noise_difference_rad=float(np.max(abs(reference[:count]-noise[:count])))))
            else:
                paired_noise[seed] = noise
        lateral = (h[:, 0] >= 25)&(h[:, 0] <= 30)
        lateral_start = int(np.searchsorted(h[:, 0], 25.))
        checks.append(dict(episode=row["episode"], samples=len(h), effort=effort,
                           measured_imu_angle_noise_rms_deg=float(np.rad2deg(np.sqrt(np.mean(noise**2)))),
                           causal_lean_estimate_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((h[available, 8]-h[available, 2])**2)))),
                           lateral_window_peak_roll_deg=float(np.max(abs(np.rad2deg(h[lateral, 1])))) if np.any(lateral) else None,
                           lateral_window_peak_world_y_displacement_m=float(np.max(abs(h[lateral, 5]-h[lateral_start, 5]))) if np.any(lateral) else None))
        native.close()
    if paired_noise or len(paired_noise_checks) != len(protocol["robust_cases"]):
        raise ValueError("Missing paired noise-stream check")
    for group in ("hybrid", "lqr", "combined"):
        subset = [r for r in rows if r["episode"].startswith(group+"/")]
        count = sum(r["success"] for r in subset)
        interval = binomtest(count, len(subset)).proportion_ci(method="wilson")
        if not np.allclose([interval.low, interval.high], summary[group]["wilson_95"]):
            raise ValueError("Independent Wilson interval does not agree")
    return dict(episodes_checked=len(checks), checks=checks, paired_noise_checks=paired_noise_checks,
                protocol_sha256=hashlib.sha256((directory/"protocol.json").read_bytes()).hexdigest())

def paired(rows):
    pairs = {}
    for row in rows:
        group = pairs.setdefault(row['case']['seed'], {})
        if row['controller'] in group:
            raise ValueError('Duplicate controller/seed')
        group[row['controller']] = row
    if any(set(p) != {'lqi','lqi_ph'} for p in pairs.values()):
        raise ValueError('Missing/duplicate matched ablation episode')
    positive = sum(p['lqi_ph']['success'] and not p['lqi']['success'] for p in pairs.values())
    negative = sum(p['lqi']['success'] and not p['lqi_ph']['success'] for p in pairs.values())
    common = [p for p in pairs.values() if p['lqi']['success'] and p['lqi_ph']['success']]
    output = dict(pairs=len(pairs), ph_only_success=positive, comparator_only_success=negative,
        mcnemar_exact_p=None if positive+negative == 0 else float(binomtest(positive,positive+negative,.5).pvalue),
        common_successes=len(common))
    rng = np.random.default_rng(918735)
    differences = np.array([int(p['lqi_ph']['success'])-int(p['lqi']['success']) for p in pairs.values()])
    sample = rng.integers(len(pairs),size=(10000,len(pairs)))
    output['success_difference'] = float(differences.mean())
    output['success_difference_bootstrap_95'] = np.quantile(differences[sample].mean(axis=1),[.025,.975]).tolist()
    if common:
        sample = rng.integers(len(common),size=(10000,len(common)))
        for metric in ('actual_wheel_effort_nm_s','actual_all_effort_nm_s','mechanical_total_positive_work_j'):
            h,l = [np.array([p[c][metric] for p in common]) for c in ('lqi_ph','lqi')]
            reduction = 1-h.mean()/l.mean()
            interval = np.quantile(1-h[sample].mean(axis=1)/l[sample].mean(axis=1),[.025,.975])
            output[metric] = dict(reduction=float(reduction),paired_bootstrap_95=interval.tolist(),
                                  ph_mean=float(h.mean()),comparator_mean=float(l.mean()))
    return output

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from experiments.upkie_stress import aggregate
    rows=json.loads((Path(args.evaluation)/'metrics.json').read_text())
    result={'summary':aggregate(rows)}
    if {r['controller'] for r in rows}=={'lqi','lqi_ph'}:result['paired']=paired(rows)
    with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
