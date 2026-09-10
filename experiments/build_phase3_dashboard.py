"""Generate the portfolio-ready Phase 3 results dashboard."""

from pathlib import Path

from src.visualization.dashboard import build_phase3_dashboard

ROOT = Path(__file__).resolve().parents[1]


def main() -> Path:
    """Build dashboard plots and Markdown from official Phase 3 results."""

    output = build_phase3_dashboard(ROOT / "results" / "phase3")
    print(f"Wrote Phase 3 dashboard to {output}")
    return output


if __name__ == "__main__":
    main()
