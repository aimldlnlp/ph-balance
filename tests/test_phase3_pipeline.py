"""Checks for the Phase 3 Hybrid PH experiment configuration."""

import unittest

from experiments.run_phase2_experiments import (
    HYBRID_PH_REFERENCE_GAINS,
    PHASE3_CONTROLLER_NAMES,
    controller_builders,
)
from src.controllers import HybridPortHamiltonianController


class Phase3PipelineTests(unittest.TestCase):
    """Keep Hybrid PH registration separate from Phase 2 baselines."""

    def test_phase2_defaults_stay_four_controller(self) -> None:
        """The existing Phase 2 builder set remains unchanged by default."""

        self.assertEqual(
            tuple(controller_builders()),
            ("PID", "PH-V1", "PH-V2", "PH-V3"),
        )

    def test_phase3_registers_hybrid_ph(self) -> None:
        """The optional Phase 3 set contains all five controllers."""

        builders = controller_builders(include_hybrid=True)

        self.assertEqual(tuple(builders), PHASE3_CONTROLLER_NAMES)
        self.assertIsInstance(
            builders["Hybrid PH"](),
            HybridPortHamiltonianController,
        )

    def test_hybrid_reference_gains_are_reproducible(self) -> None:
        """The validated Hybrid PH gain row is explicit and bounded."""

        self.assertEqual(HYBRID_PH_REFERENCE_GAINS["energy_gain"], 0.1)
        self.assertEqual(HYBRID_PH_REFERENCE_GAINS["damping_gain"], 0.01)
        self.assertEqual(len(HYBRID_PH_REFERENCE_GAINS["stabilizer_gains"]), 4)
        self.assertEqual(HYBRID_PH_REFERENCE_GAINS["torque_limit"], 3.0)


if __name__ == "__main__":
    unittest.main()
