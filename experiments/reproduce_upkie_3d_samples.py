"""Repeat fixed evaluated cases without tuning, and compare exact native arrays."""
import argparse
import json
from pathlib import Path

from experiments.evaluate_upkie_3d import hashes
from experiments.upkie_3d_pilot import Case, save_run, schedule
import numpy as np


def compare_metrics(recorded, actual):
    integrals = ("wheel_absolute_torque_integral", "all_absolute_torque_integral")
    if recorded.keys() != actual.keys() or {k: v for k, v in recorded.items() if k not in integrals} != {k: v for k, v in actual.items() if k not in integrals}:
        raise ValueError("Repeated non-integral metrics differ")
    differences = {k: actual[k]-recorded[k] for k in integrals}
    if not all(np.isclose(actual[k], recorded[k], atol=1e-12, rtol=0) for k in integrals):
        raise ValueError("Repeated integral metrics differ beyond floating-point roundoff")
    return differences


def reproduce(evaluation, output):
    evaluation, output = Path(evaluation), Path(output)
    protocol = json.loads((evaluation/"protocol.json").read_text())
    if protocol["source_sha256"] != hashes():
        raise ValueError("Frozen simulation source has changed")
    rows = json.loads((evaluation/"metrics.json").read_text())
    selected = [next(r for r in rows if r["episode"].startswith(group+"/")) for group in ("hybrid", "lqr", "combined")]
    failure = next((r for r in rows if not r["success"]), None)
    if failure and failure["episode"] not in {r["episode"] for r in selected}:
        selected.append(failure)
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    for row in selected:
        group = row["episode"].split("/")[0]
        active = row["case"]["active"]
        table = schedule(active, r=.15 if active else protocol["tuning"][group]["r"])
        target = output/row["episode"]
        actual = save_run(target, Case(**row["case"]), row["hybrid"], table)
        original = evaluation/row["episode"]
        keys = {}
        for filename in ("trajectory.npz", "observations.npz"):
            with np.load(original/filename) as before, np.load(target/filename) as after:
                if before.files != after.files:
                    raise ValueError("Reproduction schema differs")
                for key in before.files:
                    equal = np.array_equal(before[key], after[key])
                    keys[filename+":"+key] = equal
                    if not equal:
                        raise ValueError(f"Non-identical reproduction: {row['episode']} {filename}:{key}")
        recorded = json.loads((original/"metrics.json").read_text())
        differences = compare_metrics(recorded, actual)
        checks.append(dict(episode=row["episode"], arrays_identical=keys, non_integral_metrics_identical=True,
                           integral_absolute_tolerance_nm_s=1e-12, integral_differences_nm_s=differences,
                           success=actual["success"], failure=actual["failure"]))
        print(f"Identical reproduction: {row['episode']}", flush=True)
    (output/"verification.json").write_text(json.dumps(dict(cases=checks,
        scope="First sorted episode per controller/system plus first failure if different. Native/observation arrays and non-integral metrics are exact; integral reductions allow 1e-12 Nm s roundoff. This is not a second full evaluation."), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reproduce(args.evaluation, args.output)
