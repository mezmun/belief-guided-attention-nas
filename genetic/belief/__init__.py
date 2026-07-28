"""
This package contains the belief-guided NAS components.

The package is designed to keep the new belief system separate from the
existing genetic algorithm code. The public interface is exposed through
BeliefConfig and BeliefManager.
"""

from .config import BeliefConfig
from .manager import BeliefManager

__all__ = ["BeliefConfig", "BeliefManager"]
