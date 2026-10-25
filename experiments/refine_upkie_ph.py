"""Explicit PH damping refinement without modifying the archived frozen runner."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from experiments import upkie_stress as baseline
import numpy as np


def worker(output,name,case,control,table,damping):
    original=baseline.run
    try:
        baseline.run=partial(original,damping_gain=damping)
        result=baseline.worker(output,name,case,control,table)
    finally:
        baseline.run=original
    result['ph_damping_gain']=damping
    (Path(output)/name/'metrics.json').write_text(json.dumps(result,indent=2))
    return result


def source_hashes():
    result=baseline.stress_hashes()
    result['experiments/refine_upkie_ph.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def execute(output,protocol,workers=4):
    path=Path(protocol)
    declared=json.loads(path.read_text())
    if declared['source_sha256']!=source_hashes():
        raise ValueError('Refinement sources differ from declared protocol')
    if path.with_suffix('.sha256').exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Refinement protocol digest changed')
    jobs=declared['jobs']
    if workers<1 or not jobs or len({j[0] for j in jobs})!=len(jobs):
        raise ValueError('Invalid refinement jobs')
    for name,c,control,damping in jobs:
        baseline.StressCase(**c)
        if Path(name).is_absolute() or '..' in Path(name).parts or control not in ('lqi','lqi_ph') or not np.isfinite(damping) or damping<0:
            raise ValueError('Invalid refinement job')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(path,output/'protocol.json')
    for name in declared['source_sha256']:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(baseline.ROOT/name,target)
    table=baseline.integral_schedule(baseline.schedule(True,r=.6))
    np.save(output/'schedule_integral.npy',table)
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,str(output),name,c,control,table,damping) for name,c,control,damping in jobs]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            with (output/'journal.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(f"{len(rows)}/{len(jobs)} {row['episode']} pass={row['success']} failure={row['failure']} gates={row['failed_metric_gates']}",flush=True)
    if declared['source_sha256']!=source_hashes():raise ValueError('Source changed during refinement')
    rows.sort(key=lambda r:r['episode'])
    (output/'metrics.json').write_text(json.dumps(rows,indent=2))
    (output/'summary.json').write_text(json.dumps(baseline.aggregate(rows),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    execute(args.output,args.protocol,args.workers)
