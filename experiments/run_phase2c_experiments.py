"""Generate the official Phase 2C controller comparison artifacts."""

from __future__ import annotations

from pathlib import Path

from experiments.run_phase2_experiments import ROOT, main as run_experiments


def main() -> Path:
    """Run the shared experiment pipeline into ``results/phase2c``."""

    return run_experiments(
        output_root=ROOT / "results" / "phase2c",
        summary_title="PH-Balance Phase 2C experiment summary",
    )


if __name__ == "__main__":
    main()
