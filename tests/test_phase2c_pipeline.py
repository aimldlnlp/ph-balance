"""Checks for the official Phase 2C experiment configuration."""

import unittest

from experiments.run_phase2_experiments import (
    CONTROLLER_NAMES,
    PH_V3_REFERENCE_GAINS,
    all_experiment_specs,
    controller_builders,
)


class Phase2CPipelineTests(unittest.TestCase):
    """Keep Phase 2C scenarios and controller coverage explicit."""

    def test_all_required_controllers_are_registered(self) -> None:
        """The official comparison contains all four controllers."""

        self.assertEqual(tuple(controller_builders()), CONTROLLER_NAMES)
        self.assertEqual(
            set(controller_builders()),
            {"PID", "PH-V1", "PH-V2", "PH-V3"},
        )
        for builder in controller_builders().values():
            self.assertIsNotNone(builder())

    def test_phase2c_scenario_matrix_is_complete(self) -> None:
        """The official matrix has three main and nine sweep scenarios."""

        specs = all_experiment_specs()

        self.assertEqual(len(specs), 12)
        self.assertEqual(
            {spec.group for spec in specs},
            {"main", "angle_sweep", "disturbance_sweep"},
        )
        self.assertEqual(
            sum(spec.group == "main" for spec in specs),
            3,
        )
        self.assertEqual(
            sum(spec.group == "angle_sweep" for spec in specs),
            4,
        )
        self.assertEqual(
            sum(spec.group == "disturbance_sweep" for spec in specs),
            5,
        )

    def test_ph_v3_reference_gains_are_configured(self) -> None:
        """PH-V3 has a fixed, reproducible non-optimized baseline."""

        self.assertEqual(PH_V3_REFERENCE_GAINS["energy_gain"], 0.1)
        self.assertEqual(PH_V3_REFERENCE_GAINS["angle_gain"], 4.0)
        self.assertEqual(PH_V3_REFERENCE_GAINS["position_gain"], 0.1)
        self.assertEqual(PH_V3_REFERENCE_GAINS["damping_gain"], 0.01)
        self.assertEqual(PH_V3_REFERENCE_GAINS["torque_limit"], 3.0)


if __name__ == "__main__":
    unittest.main()
