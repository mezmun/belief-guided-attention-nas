"""
This module measures how different a candidate is from the evaluated archive.

Novelty is kept separate from uncertainty. It is used only for exploration, so
an unknown architecture is not counted twice through both uncertainty and
novelty bonuses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from .archive import EvaluatedArchitectureArchive
from .encoder import ArchitectureEncoding
from .similarity import ArchitectureSimilarity


@dataclass(frozen=True)
class NoveltyEstimate:
    """Store archive-based novelty values for one candidate."""

    novelty: float
    max_similarity: float
    mean_top_k_similarity: float
    neighbour_count: int

    def to_dict(self) -> Dict[str, object]:
        """Return novelty information as a plain dictionary."""

        return asdict(self)


class ArchitectureNovelty:
    """Calculate novelty from the nearest evaluated architectures."""

    def __init__(self, similarity: ArchitectureSimilarity, top_k: int = 5) -> None:
        """Create the novelty calculator."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.similarity = similarity
        self.top_k = int(top_k)

    def estimate(
        self,
        candidate: ArchitectureEncoding,
        archive: EvaluatedArchitectureArchive,
    ) -> NoveltyEstimate:
        """Return one minus the mean similarity of the nearest archive items."""

        values = [
            self.similarity.compare(candidate, entry.encoding).total
            for entry in archive.entries()
            if entry.architecture_id != candidate.architecture_id
        ]
        if not values:
            return NoveltyEstimate(
                novelty=1.0,
                max_similarity=0.0,
                mean_top_k_similarity=0.0,
                neighbour_count=0,
            )

        ordered = sorted(values, reverse=True)
        selected = ordered[: min(self.top_k, len(ordered))]
        mean_similarity = sum(selected) / len(selected)
        return NoveltyEstimate(
            novelty=float(max(0.0, min(1.0, 1.0 - mean_similarity))),
            max_similarity=float(ordered[0]),
            mean_top_k_similarity=float(mean_similarity),
            neighbour_count=len(selected),
        )
