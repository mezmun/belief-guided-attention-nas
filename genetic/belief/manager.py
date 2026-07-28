"""
This module provides the main entry point for the belief-guided NAS system.

The first version only loads the configuration and reports the active mode.
Later versions will connect the archive, similarity, belief, uncertainty,
and selection modules through this class.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import BeliefConfig


class BeliefManager:
    """Coordinate belief-related components through one public interface."""

    def __init__(self, config: Optional[BeliefConfig] = None) -> None:
        """Make the manager with a validated configuration."""

        self.config = config or BeliefConfig.from_ini()

    @property
    def is_enabled(self) -> bool:
        """Return True when the belief system is enabled."""

        return self.config.enabled

    @property
    def is_monitoring(self) -> bool:
        """Return True when belief scores are recorded without guiding selection."""

        return self.is_enabled and self.config.mode == "monitor"

    @property
    def is_guided(self) -> bool:
        """Return True when belief scores guide offspring evaluation."""

        return self.is_enabled and self.config.mode == "guided"

    def describe(self) -> Dict[str, Any]:
        """Return a small status summary for logs and debugging."""

        return {
            "enabled": self.is_enabled,
            "mode": self.config.mode,
            "warmup_generations": self.config.warmup_generations,
            "candidate_multiplier": self.config.candidate_multiplier,
            "evaluation_budget": self.config.evaluation_budget,
        }


if __name__ == "__main__":
    manager = BeliefManager()
    print("Belief manager was created successfully.")
    for key, value in manager.describe().items():
        print(f"{key}: {value}")
