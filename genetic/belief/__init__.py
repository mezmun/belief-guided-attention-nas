"""
This package contains the belief-guided NAS components.

The package keeps the new belief system separate from the existing genetic
algorithm code. The public interface is exposed through BeliefConfig,
BeliefManager, ArchitectureEncoder, and ArchitectureEncoding.
"""

from .config import BeliefConfig
from .encoder import ArchitectureEncoder, ArchitectureEncoding
from .manager import BeliefManager

__all__ = [
    "ArchitectureEncoder",
    "ArchitectureEncoding",
    "BeliefConfig",
    "BeliefManager",
]
