"""Repeat fixed success/failure recordings with immutable simulation sources."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
from experiments.upkie_stress import stress_hashes, worker
from experiments.refine_upkie_ph import source_hashes as refinement_hashes, worker as refinement_worker
import numpy as np


def reproduce(evaluation,output):
    evaluation,output=Path(evaluation),Path(output)
    manifest=json.loads((evaluation/'protocol.json').read_text())
    refined=any(len(job)==4 for job in manifest['jobs'])
    if manifest['source_sha256'] != (refinement_hashes() if refined else stress_hashes()):
        raise ValueError('Evaluation sources changed')
    jobs=manifest['jobs']
    selected=[]
    for controller in ('lqi','lqi_ph'):
        for success in (True,False):
            for job in sorted(jobs):
                name,c,control=job[:3]
                path=evaluation/name/'metrics.json'
                if control==controller and path.exists():
                    row=json.loads(path.read_text())
                    if row['success']==success:
                        selected.append(job);break
    if len(selected)<2:
        for job in sorted(jobs):
            if job not in selected and (evaluation/job[0]/'metrics.json').exists():
                selected.append(job);break
    if len(selected)<2:
        raise ValueError('Need at least two recorded examples before reproduction')
    output.mkdir(parents=True,exist_ok=False)
    tables={job[2]:np.load(evaluation/'schedule_integral.npy') for job in selected}
    checks=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(refinement_worker if refined else worker,str(output),job[0],job[1],job[2],tables[job[2]],*job[3:]) for job in selected]
        for job,future in zip(selected,futures):
            name=job[0]
            repeated=future.result()
            for filename in ('trajectory.npz','observations.npz'):
                with np.load(evaluation/name/filename) as before,np.load(output/name/filename) as after:
                    if set(before.files)!=set(after.files) or any(not np.array_equal(before[k],after[k]) for k in before.files):
                        raise ValueError('Native or observation arrays differ: '+name)
            original=json.loads((evaluation/name/'metrics.json').read_text())
            for key in original:
                if key=='wall_seconds':continue
                left,right=original[key],repeated[key]
                if isinstance(left,(int,float)) and not isinstance(left,bool):
                    equal=np.isclose(left,right,atol=1e-10,rtol=0)
                elif isinstance(left,list) and left and all(isinstance(x,(int,float)) for x in left):
                    equal=np.allclose(left,right,atol=1e-10,rtol=0)
                else:equal=left==right
                if not equal:raise ValueError(f'Metric differs: {name}/{key}')
            checks.append(dict(episode=name,native_and_observations_bitwise_identical=True,
                               metrics_absolute_tolerance=1e-10,success=repeated['success'],failure=repeated['failure']))
            print('Exact native reproduction: '+name,flush=True)
    (output/'verification.json').write_text(json.dumps(dict(cases=checks,scope='Fixed examples, not a second full evaluation.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    reproduce(args.evaluation,args.output)
