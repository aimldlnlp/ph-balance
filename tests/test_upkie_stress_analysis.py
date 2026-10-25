"""Failure-aware paired comparisons and fresh-population protocol guards."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import importlib.util
from unittest.mock import patch
HAS_MUJOCO = importlib.util.find_spec('mujoco') is not None
if HAS_MUJOCO:
    from experiments.analyze_results import paired
    from experiments.freeze_upkie_stress import freeze
    from experiments import refine_upkie_ph
    from experiments.plan_upkie_refinement import plan


@unittest.skipUnless(HAS_MUJOCO, 'Install MuJoCo for stress evidence checks')
class StressAnalysisTests(unittest.TestCase):
    def row(self,seed,controller,success,effort):
        return dict(case=dict(seed=seed),controller=controller,success=success,
                    actual_wheel_effort_nm_s=effort,actual_all_effort_nm_s=effort,
                    mechanical_total_positive_work_j=effort)

    def test_failed_exposure_does_not_create_efficiency_claim(self):
        rows=[self.row(1,'lqi_ph',True,8),self.row(1,'lqi',True,10),
              self.row(2,'lqi_ph',True,100),self.row(2,'lqi',False,.01)]
        result=paired(rows)
        self.assertEqual(result['common_successes'],1)
        self.assertEqual(result['ph_only_success'],1)
        self.assertAlmostEqual(result['actual_wheel_effort_nm_s']['reduction'],.2)
        self.assertEqual(result['success_difference'],.5)

    def test_incomplete_and_duplicate_pairs_are_rejected(self):
        row=self.row(1,'lqi_ph',True,1)
        for rows in ([row],[row,row]):
            with self.assertRaises(ValueError):paired(rows)

    def test_freeze_rejects_failed_pilot_before_writing(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'pilot';pilot.mkdir()
            (pilot/'protocol.json').write_text(json.dumps(dict(source_sha256={})))
            (pilot/'metrics.json').write_text(json.dumps([dict(controller='lqi_ph',success=False)]*16))
            with patch('experiments.freeze_upkie_stress.stress_hashes',return_value={}):
                with self.assertRaises(ValueError):freeze(pilot,root/'frozen')
            self.assertFalse((root/'frozen').exists())

    def test_refinement_restores_frozen_runner_after_exception(self):
        original=refine_upkie_ph.baseline.run
        with patch.object(refine_upkie_ph.baseline,'worker',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                refine_upkie_ph.worker('unused','case',{},'lqi_ph',None,.04)
        self.assertIs(refine_upkie_ph.baseline.run,original)

    def test_followup_rejects_pilot_below_declared_gate(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'pilot';pilot.mkdir()
            (pilot/'protocol.json').write_text(json.dumps(dict(source_sha256={})))
            rows=[dict(success=i<30,ph_damping_gain=.02) for i in range(32)]
            (pilot/'metrics.json').write_text(json.dumps(rows))
            with patch('experiments.plan_upkie_refinement.source_hashes',return_value={}):
                with self.assertRaises(ValueError):plan('freeze',pilot,root/'frozen')
            self.assertFalse((root/'frozen').exists())


if __name__=='__main__':unittest.main()
