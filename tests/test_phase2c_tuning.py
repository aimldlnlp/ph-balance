"""Checks for the reproducible PH-V3 gain-search configuration."""

import unittest

from experiments.tune_phase2c_ph_v3 import GainCandidate, gain_grid


class Phase2CTuningTests(unittest.TestCase):
    """Ensure the search is a complete, deterministic Cartesian grid."""

    def test_gain_grid_cardinality_and_uniqueness(self) -> None:
        """Every configured gain combination appears exactly once."""

        candidates = gain_grid()
        keys = {
            (
                candidate.energy_gain,
                candidate.angle_gain,
                candidate.position_gain,
                candidate.damping_gain,
            )
            for candidate in candidates
        }

        self.assertEqual(len(candidates), 1400)
        self.assertEqual(len(keys), len(candidates))

    def test_candidate_builds_ph_v3(self) -> None:
        """A grid candidate maps to the existing PH-V3 interface."""

        candidate = GainCandidate(0.1, 4.0, 0.1, 0.01)
        controller = candidate.build_controller()

        self.assertEqual(controller.energy_gain, 0.1)
        self.assertEqual(controller.angle_gain, 4.0)
        self.assertEqual(controller.position_gain, 0.1)
        self.assertEqual(controller.damping_gain, 0.01)


if __name__ == "__main__":
    unittest.main()
