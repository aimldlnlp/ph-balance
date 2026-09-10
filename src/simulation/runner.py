"""Compatibility exports for the simulation framework."""

from .simulator import SimulationConfig, SimulationResult, Simulator

SimulationRunner = Simulator

__all__ = ["SimulationConfig", "SimulationResult", "SimulationRunner"]
