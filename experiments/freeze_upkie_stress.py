"""Freeze an explicit fresh held-out population after successful development."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.upkie_stress import StressCase, stress_hashes


def freeze(pilot, output, seed_start=480000):
    pilot,output = Path(pilot),Path(output)
    metadata=json.loads((pilot/'protocol.json').read_text())
    if metadata['source_sha256'] != stress_hashes():
        raise ValueError('Pilot used different sources; rerun development')
    rows=json.loads((pilot/'metrics.json').read_text())
    primary=[r for r in rows if r['controller']=='lqi_ph']
    if len(primary)<16 or sum(r['success'] for r in primary)/len(primary)<.95:
        raise ValueError('Primary pilot gate not met')
    seeds=set(range(seed_start,seed_start+200))
    for path in output.parent.glob('*/protocol.json'):
        previous=json.loads(path.read_text())
        used={job[1]['seed'] for job in previous.get('jobs',[])}
        if used & seeds:
            raise ValueError('Held-out seeds already used or frozen: '+str(path))
    jobs=[]
    for seed in sorted(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([seed,12785]))
        sign=float(rng.choice([-1.,1.]))
        c=StressCase(seed=seed,active=True,pitch=float(rng.uniform(-10,10)),
            heading=float(rng.choice([-15,15])),torso_scale=float(rng.uniform(.8,1.2)),
            friction=float(rng.uniform(.4,.8)),noise_deg=float(rng.uniform(.2,.5)),
            pulse=float(rng.uniform(.4,.8)),lateral=sign*3.,delay=float(rng.uniform(.01,.025)),
            jitter=.005,dropout=.1,motor_tau=float(rng.uniform(.005,.02)),
            pitch_bias_deg=float(rng.uniform(-.25,.25)),gyro_bias_deg_s=float(rng.uniform(-.1,.1)),
            heading_drift_deg_s=float(rng.uniform(-.01,.01)),persistent_torque=sign*.1,warmup=.2)
        for control in ('lqi_ph','lqi'):
            jobs.append((f'combined/{control}/{seed}',asdict(c),control))
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(version=1,jobs=jobs,source_sha256=stress_hashes(),
        pilot_sha256=hashlib.sha256((pilot/'metrics.json').read_bytes()).hexdigest(),
        freeze_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary='lqi_ph',primary_target=dict(episodes=200,successes_at_least=190,horizon_s=60),
        design='Scale nominal multibody LQR input cost by 100; add six bounded coordinate integrals with clipping/slew antiwindup. Same design in both arms; only PH energy/port terms differ (0.1, 0.01).',
        interpretation='Matched PH ablation, not best-tuned LQI superiority. Failure exposure is separate. Positive mechanical work is not electrical energy.',
        acceptance='Every disturbance has its own <=3 s recovery into +/-2 deg with >=5 s hold before next disturbance. Full 60 s, no nonwheel floor contact, source torque/speed/position bounds, final XY<=0.15m, excursion<=0.50m, heading<=5deg, three 4cm height cycles, settled lean<=2deg, settled height error<=1cm.',
        windows='Exclude release 0-3s; persistent onset 8-11s; pitch 15-18.2s and 35-38.2s; lateral 25-28.2s; height ramps 6,14,22,30,38,46 through +4s. Persistent torque starts at8s; lateral ends25.2s.',
        assumptions='Fused IMU with heading reference; 0.2s pre-release static sensor warmup, no fixture after0s; uncalibrated first-order motor lag; no raw accelerometer fusion, electrical losses or global passivity proof.',
        analysis='Wilson 95% success intervals, exact paired McNemar test, 10000-draw paired bootstrap; common-success mechanical work/actual-torque effort plus all failures. No final-seed tuning.')
    path=output/'protocol.json'
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    print(f'Frozen {len(jobs)} episodes on {len(seeds)} paired fresh seeds: {path}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed-start',type=int,default=480000)
    args=parser.parse_args()
    freeze(args.pilot,args.output,args.seed_start)
