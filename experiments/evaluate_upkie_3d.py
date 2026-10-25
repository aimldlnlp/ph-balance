"""Versioned CPU evaluation: development seeds first, immutable held-out freeze.

Every episode retains native state, observations/estimates, controls and model.
No failed run is discarded and output directories cannot be overwritten.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Avoid oversubscription in CPU workers (set before importing NumPy/SciPy).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
import scipy

from experiments.upkie_3d_pilot import Case, save_run, schedule
from src.simulation.upkie import ROOT, SOURCE_REVISION

SIMULATION_FILES = ["src/simulation/upkie.py", "src/simulation/upkie_3d.py", "src/simulation/upkie_observer.py",
                    "src/simulation/upkie_active_lqr.py", "experiments/upkie_3d_pilot.py", "experiments/evaluate_upkie_3d.py"]


def hashes():
    files = [ROOT/f for f in SIMULATION_FILES] + list((ROOT/"assets/upkie").rglob("*"))
    return {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}


def cases(seeds, active=False):
    result = []
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 89123]))
        uncertainty = .1 if active else .2
        result.append(Case(seed=seed, pitch=float(rng.uniform(-10, 10)), heading=float(rng.choice([-15., 15.])),
                           torso_scale=float(rng.uniform(1-uncertainty, 1+uncertainty)),
                           friction=float(rng.uniform(.6, 1.)), noise_deg=.2 if active else float(rng.uniform(.2, .5)),
                           active=active, pulse=float(rng.uniform(.4, .8)), lateral=float(rng.choice([-1., 1.]))))
    return result


def wilson(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def worker(output, case_dict, hybrid, table):
    start = time.perf_counter()
    metrics = save_run(output, Case(**case_dict), hybrid, np.asarray(table))
    metrics["wall_seconds"] = time.perf_counter()-start
    return metrics


def batch(output, jobs, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, str(output/name), asdict(case), hybrid, table): name for name, case, hybrid, table in jobs}
        for future in as_completed(futures):
            row = future.result()
            row["episode"] = futures[future]
            rows.append(row)
            # Incremental journal survives an interrupted evaluation.
            with (output/"journal.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
            print(f"{len(rows)}/{len(jobs)} {row['episode']} pass={row['success']} failure={row['failure']}", flush=True)
    return sorted(rows, key=lambda r: r["episode"])


def summary(rows):
    result = {}
    for group in ("hybrid", "lqr", "combined"):
        group_rows = [r for r in rows if r["episode"].startswith(group+"/")]
        if not group_rows:
            continue
        n = len(group_rows)
        successes = sum(r["success"] for r in group_rows)
        failures = {}
        for row in group_rows:
            if not row["success"]:
                key = row["failure"] or "metric_gate"
                failures[key] = failures.get(key, 0)+1
        effort = [r["wheel_absolute_torque_integral"] for r in group_rows if r["success"]]
        result[group] = dict(episodes=n, successes=successes, success_rate=successes/n,
                             wilson_95=wilson(successes, n), failures=failures,
                             successful_mean_wheel_effort=None if not effort else float(np.mean(effort)))
    if "hybrid" in result and "lqr" in result:
        pairs = {}
        for row in rows:
            if row["episode"].startswith(("hybrid/", "lqr/")):
                pairs.setdefault(row["case"]["seed"], {})["hybrid" if row["hybrid"] else "lqr"] = row
        common = [p for p in pairs.values() if len(p) == 2 and all(r["success"] for r in p.values())]
        if common:
            h = np.array([p["hybrid"]["wheel_absolute_torque_integral"] for p in common])
            l = np.array([p["lqr"]["wheel_absolute_torque_integral"] for p in common])
            # Paired bootstrap resamples episodes, retaining pairing within each draw.
            rng = np.random.default_rng(456789)
            sample = rng.integers(len(h), size=(10000, len(h)))
            bootstrap = 1-np.mean(h[sample], axis=1)/np.mean(l[sample], axis=1)
            reduction = float(1-np.mean(h)/np.mean(l))
            result["ablation"] = dict(common_success_pairs=len(common), wheel_effort_reduction=reduction,
                                      paired_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(),
                                      point_hypothesis_met=bool(reduction >= .1 and result["hybrid"]["successes"] >= result["lqr"]["successes"]),
                                      caveat="Effort restricted to paired successes; failures and completed horizons reported separately. No global stability claim.")
    return result


def write_results(output, rows):
    (output/"metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary(rows), indent=2), encoding="utf-8")
    if rows:
        flat = [{**{k: v for k, v in r.items() if k != "case"}, **r["case"]} for r in rows]
        with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tune", "pilot", "freeze", "final"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--protocol")
    parser.add_argument("--tuning")
    parser.add_argument("--pilot")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"run_metadata.json").write_text(json.dumps(dict(mode=args.mode, source_sha256=hashes(),
        python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__, scipy=scipy.__version__,
        source_revision=SOURCE_REVISION, argv=sys.argv), indent=2), encoding="utf-8")
    if args.mode == "freeze":
        pilot_path = Path(args.pilot)
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        for group, minimum in (("locked_pilot", .95), ("active_pilot", .9)):
            subset = [row for row in pilot if row["episode"].startswith(group+"/")]
            if len(subset) < 16 or sum(row["success"] for row in subset)/len(subset) < minimum:
                raise ValueError(f"Pilot gate not met: {group}; investigate before freeze")
        tuning = json.loads(Path(args.tuning).read_text(encoding="utf-8"))
        for path in (pilot_path.parent, Path(args.tuning).parent):
            metadata = json.loads((path/"run_metadata.json").read_text(encoding="utf-8"))
            if metadata["source_sha256"] != hashes():
                raise ValueError(f"Development evidence used different sources: {path}; rerun pilot/tuning before freeze")
        protocol = dict(version=1, source_revision=SOURCE_REVISION, source_sha256=hashes(),
                        versions=dict(python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        tuning=tuning, robust_cases=[asdict(c) for c in cases(range(100000, 100200))],
                        combined_cases=[asdict(c) for c in cases(range(200000, 200100), True)],
                        locked_validation=[asdict(Case(seed=300000+i, pitch=pitch, heading=heading, lateral=1. if heading>0 else -1.))
                                           for i, (pitch, heading) in enumerate(((-10., -15.), (-10., 15.), (10., -15.), (10., 15.)))],
                        pilot_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
                        observer_delay_s=.01, robust_success_target=.95, combined_success_target=.90,
                        limits="Source URDF torque, joint velocity and position; any violation terminates as failure.",
                        success="60 seconds, no nonwheel floor contact, settling <=3 seconds into +/-2 degrees after release and pitch pulses (>=5s persistent hold in each event segment), final wheel-midpoint xy displacement <=0.15m, max <=0.50m, heading error <=5 degrees. Combined additionally settled lean <=2deg and torso height error <=0.01m over three 0.04m cycles.",
                        evaluation="All scheduled episodes in denominator; Wilson 95% binomial interval. Paired successful-run wheel effort separated from failures; paired 10000-draw bootstrap; no global stability proof.")
        (output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        (output/"protocol.sha256").write_text(hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest(), encoding="ascii")
        return
    if args.mode == "final":
        protocol_path = Path(args.protocol)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = (protocol_path.parent/"protocol.sha256").read_text(encoding="ascii")
        if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != expected or protocol["source_sha256"] != hashes():
            raise ValueError("Frozen protocol or simulation source changed; create a new protocol with fresh held-out seeds")
        (output/"protocol.json").write_bytes(protocol_path.read_bytes())
        jobs = []
        for group, hybrid in (("hybrid", True), ("lqr", False)):
            table = schedule(r=protocol["tuning"][group]["r"])
            jobs.extend((f"{group}/{c['seed']}", Case(**c), hybrid, table) for c in protocol["robust_cases"])
        table = schedule(True)
        jobs.extend((f"combined/{c['seed']}", Case(**c), True, table) for c in protocol["combined_cases"])
        table = schedule(r=protocol["tuning"]["hybrid"]["r"])
        jobs.extend((f"locked_validation/{c['seed']}", Case(**c), True, table) for c in protocol["locked_validation"])
    elif args.mode == "tune":
        # Equal candidate and episode budgets; PH gains fixed before tuning.
        jobs = []
        for r in (.15, .3, .6):
            table = schedule(r=r)
            for hybrid in (True, False):
                group = "hybrid" if hybrid else "lqr"
                jobs.extend((f"{group}_r{r}/{c.seed}", c, hybrid, table) for c in cases(range(1000, 1008)))
    else:
        jobs = []
        for active in (False, True):
            table = schedule(active)
            group = "active_pilot" if active else "locked_pilot"
            jobs.extend((f"{group}/{c.seed}", c, True, table) for c in cases(range(2000, 2016), active))
    rows = batch(output, jobs, args.workers)
    write_results(output, rows)
    if args.mode == "tune":
        selected = {}
        for group in ("hybrid", "lqr"):
            ranked = []
            for r in (.15, .3, .6):
                subset = [row for row in rows if row["episode"].startswith(f"{group}_r{r}/")]
                good = [row["wheel_absolute_torque_integral"] for row in subset if row["success"]]
                ranked.append((sum(row["success"] for row in subset), -float(np.mean(good)) if good else -np.inf, r))
            best = max(ranked)
            selected[group] = dict(r=best[2], tuning_successes=best[0], candidates=3, episodes_per_candidate=8,
                                   seed_range=[1000, 1007], energy_gain=.1 if group == "hybrid" else 0., damping_gain=.01 if group == "hybrid" else 0.)
        (output/"selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
