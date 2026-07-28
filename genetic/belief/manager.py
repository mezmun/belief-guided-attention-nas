"""
This module provides the main entry point for the belief-guided NAS system.

The manager keeps the new belief components behind one public interface. The
current version loads the configuration and exposes the architecture encoder.
It still does not change the genetic algorithm behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .config import BeliefConfig
from .encoder import ArchitectureEncoder, ArchitectureEncoding


class BeliefManager:
    """Coordinate belief-related components through one public interface."""

    def __init__(self, config: Optional[BeliefConfig] = None) -> None:
        """Create the manager with a validated configuration."""

        self.config = config or BeliefConfig.from_ini()
        self.encoder = ArchitectureEncoder()

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

    def encode_architecture(self, individual: Any) -> ArchitectureEncoding:
        """Encode one architecture through the package public interface."""

        return self.encoder.encode(individual)

    def encode_architectures(self, individuals: Iterable[Any]) -> List[ArchitectureEncoding]:
        """Encode several architectures through the package public interface."""

        return self.encoder.encode_many(individuals)

    def describe(self) -> Dict[str, Any]:
        """Return a small status summary for logs and debugging."""

        return {
            "enabled": self.is_enabled,
            "mode": self.config.mode,
            "warmup_generations": self.config.warmup_generations,
            "candidate_multiplier": self.config.candidate_multiplier,
            "evaluation_budget": self.config.evaluation_budget,
            "encoder_version": self.encoder.VERSION,
        }


if __name__ == "__main__":
    manager = BeliefManager()
    print("Belief manager was created successfully.")
    for key, value in manager.describe().items():
        print(f"{key}: {value}")
