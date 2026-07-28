"""
This package contains the belief-guided NAS components.

The package keeps the new method separate from the existing genetic algorithm.
BeliefManager is the main public integration point used by evolve.py.
"""

from .config import BeliefConfig
from .encoder import ArchitectureEncoder, ArchitectureEncoding
from .manager import BeliefManager, CyclePreparation

__all__ = [
    "ArchitectureEncoder",
    "ArchitectureEncoding",
    "BeliefConfig",
    "BeliefManager",
    "CyclePreparation",
]
