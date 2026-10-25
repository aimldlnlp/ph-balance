"""Predeclare PH-only follow-up after the archived matched-ablation experiment."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.upkie_stress import StressCase
from experiments.refine_upkie_ph import source_hashes


def random_case(seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,12785]))
    sign=float(rng.choice([-1.,1.]))
    return StressCase(seed=seed,active=True,pitch=float(rng.uniform(-10,10)),
        heading=float(rng.choice([-15,15])),torso_scale=float(rng.uniform(.8,1.2)),
        friction=float(rng.uniform(.4,.8)),noise_deg=float(rng.uniform(.2,.5)),
        pulse=float(rng.uniform(.4,.8)),lateral=sign*3.,delay=float(rng.uniform(.01,.025)),
        jitter=.005,dropout=.1,motor_tau=float(rng.uniform(.005,.02)),
        pitch_bias_deg=float(rng.uniform(-.25,.25)),gyro_bias_deg_s=float(rng.uniform(-.1,.1)),
        heading_drift_deg_s=float(rng.uniform(-.01,.01)),persistent_torque=sign*.1,warmup=.2)


def plan(mode,evidence,output):
    evidence,output=Path(evidence),Path(output)
    rows=json.loads((evidence/'metrics.json').read_text())
    if json.loads((evidence/'protocol.json').read_text())['source_sha256']!=source_hashes():
        raise ValueError('Development sources differ')
    if mode=='pilot':
        candidates={d:[r for r in rows if r['ph_damping_gain']==d] for d in (.01,.02,.04,.08)}
        if any(len(r)!=8 for r in candidates.values()):raise ValueError('Incomplete equal-budget damping development')
        damping=max(candidates,key=lambda d:(sum(r['success'] for r in candidates[d]),-d))
        seeds=range(493000,493032)
    else:
        if len(rows)!=32 or sum(r['success'] for r in rows)<31:
            raise ValueError('Follow-up pilot gate not met: require >=31/32')
        gains={r['ph_damping_gain'] for r in rows}
        if len(gains)!=1:raise ValueError('Mixed pilot gains')
        damping=gains.pop();seeds=range(494000,494200)
    wanted=set(seeds)
    for path in output.parent.glob('*/protocol.json'):
        used={job[1]['seed'] for job in json.loads(path.read_text()).get('jobs',[])}
        if used&wanted:raise ValueError('Follow-up seeds already used: '+str(path))
    jobs=[(f'combined/lqi_ph/{seed}',asdict(random_case(seed)),'lqi_ph',damping) for seed in seeds]
    output.mkdir(parents=True,exist_ok=False)
    declaration=dict(mode=mode,jobs=jobs,source_sha256=source_hashes(),
        evidence_sha256=hashlib.sha256((evidence/'metrics.json').read_bytes()).hexdigest(),
        ph_energy_gain=.1,ph_damping_gain=damping,primary_successes_at_least=31 if mode=='pilot' else 190,
        scope='Same task, ranges, source limits and event-aware acceptance as stress v1. Only PH damping gain changes. PH-only follow-up; prior matched ablation remains separately archived. No direct paired comparison between v1 and v2.',
        selection='Four damping gains x eight declared extreme development cases, select most successes then smallest damping. Independent 32-case random pilot requires >=31 successes before 200 fresh held-out cases.')
    path=output/'protocol.json';path.write_text(json.dumps(declaration,indent=2))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    print(f'{mode}: {len(jobs)} cases; fixed PH damping {damping}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('pilot','freeze'))
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();plan(args.mode,args.evidence,args.output)
