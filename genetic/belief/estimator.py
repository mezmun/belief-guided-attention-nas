"""
This module propagates known fitness values through architecture similarity.

It calculates a belief mean for an unevaluated candidate by combining all
archive fitness values with kernel weights. It also reports evidence strength,
effective neighbour count, and neighbour disagreement for later uncertainty
calibration.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from .archive import ArchiveEntry, EvaluatedArchitectureArchive
from .encoder import ArchitectureEncoding
from .similarity import ArchitectureSimilarity


@dataclass(frozen=True)
class NeighbourEvidence:
    """Describe the contribution of one evaluated architecture."""

    architecture_id: str
    fitness: float
    similarity: float
    kernel_weight: float
    normalized_weight: float
    weighted_contribution: float

    def to_dict(self) -> Dict[str, float | str]:
        """Return neighbour evidence as a plain dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class BeliefEstimate:
    """Store the pre-evaluation belief information for one candidate."""

    architecture_id: str
    belief_mean: float
    evidence_strength: float
    effective_neighbour_count: float
    neighbour_disagreement: float
    max_similarity: float
    used_neighbour_count: int
    excluded_exact_match_count: int
    used_prior_only: bool
    neighbours: List[NeighbourEvidence]

    def to_dict(self) -> Dict[str, object]:
        """Return the complete estimate as a serializable dictionary."""

        data = asdict(self)
        data["neighbours"] = [item.to_dict() for item in self.neighbours]
        return data


class SimilarityBeliefEstimator:
    """Calculate archive-wide kernel-weighted fitness beliefs."""

    VERSION = "1.0"

    def __init__(
        self,
        similarity: Optional[ArchitectureSimilarity] = None,
        kernel_bandwidth: float = 0.25,
        minimum_kernel_weight: float = 1e-12,
    ) -> None:
        """Create the estimator and validate its kernel settings."""

        if kernel_bandwidth <= 0 or not math.isfinite(kernel_bandwidth):
            raise ValueError("kernel_bandwidth must be finite and greater than zero")
        if minimum_kernel_weight < 0 or not math.isfinite(minimum_kernel_weight):
            raise ValueError("minimum_kernel_weight must be finite and non-negative")

        self.similarity = similarity or ArchitectureSimilarity()
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.minimum_kernel_weight = float(minimum_kernel_weight)

    def estimate_one(
        self,
        candidate: ArchitectureEncoding,
        archive: EvaluatedArchitectureArchive,
        top_neighbours: int = 5,
        exclude_exact_match: bool = True,
    ) -> BeliefEstimate:
        """Estimate one candidate from all eligible entries in the archive."""

        if len(archive) == 0:
            raise ValueError("Belief estimation requires at least one archive entry")
        if top_neighbours < 0:
            raise ValueError("top_neighbours must be zero or greater")

        weighted_items: List[tuple[ArchiveEntry, float, float]] = []
        excluded_exact_match_count = 0

        for entry in archive.entries():
            if exclude_exact_match and entry.architecture_id == candidate.architecture_id:
                excluded_exact_match_count += 1
                continue

            similarity_value = self.similarity.compare(candidate, entry.encoding).total
            kernel_weight = self.kernel_weight(similarity_value)
            if kernel_weight >= self.minimum_kernel_weight:
                weighted_items.append((entry, similarity_value, kernel_weight))

        prior_mean = self._archive_prior_mean(archive, candidate, exclude_exact_match)
        if not weighted_items:
            return BeliefEstimate(
                architecture_id=candidate.architecture_id,
                belief_mean=prior_mean,
                evidence_strength=0.0,
                effective_neighbour_count=0.0,
                neighbour_disagreement=0.0,
                max_similarity=0.0,
                used_neighbour_count=0,
                excluded_exact_match_count=excluded_exact_match_count,
                used_prior_only=True,
                neighbours=[],
            )

        total_weight = sum(item[2] for item in weighted_items)
        belief_mean = sum(
            entry.fitness_mean * kernel_weight
            for entry, _, kernel_weight in weighted_items
        ) / total_weight

        disagreement = sum(
            kernel_weight * (entry.fitness_mean - belief_mean) ** 2
            for entry, _, kernel_weight in weighted_items
        ) / total_weight

        squared_weight_sum = sum(item[2] ** 2 for item in weighted_items)
        effective_count = (
            total_weight**2 / squared_weight_sum if squared_weight_sum > 0 else 0.0
        )

        sorted_items = sorted(weighted_items, key=lambda item: item[2], reverse=True)
        selected_items = sorted_items[:top_neighbours] if top_neighbours else []
        neighbours = [
            NeighbourEvidence(
                architecture_id=entry.architecture_id,
                fitness=entry.fitness_mean,
                similarity=similarity_value,
                kernel_weight=kernel_weight,
                normalized_weight=kernel_weight / total_weight,
                weighted_contribution=(kernel_weight / total_weight) * entry.fitness_mean,
            )
            for entry, similarity_value, kernel_weight in selected_items
        ]

        return BeliefEstimate(
            architecture_id=candidate.architecture_id,
            belief_mean=float(belief_mean),
            evidence_strength=float(total_weight),
            effective_neighbour_count=float(effective_count),
            neighbour_disagreement=float(disagreement),
            max_similarity=max(item[1] for item in weighted_items),
            used_neighbour_count=len(weighted_items),
            excluded_exact_match_count=excluded_exact_match_count,
            used_prior_only=False,
            neighbours=neighbours,
        )

    def estimate_many(
        self,
        candidates: Iterable[ArchitectureEncoding],
        archive: EvaluatedArchitectureArchive,
        top_neighbours: int = 5,
        exclude_exact_match: bool = True,
    ) -> List[BeliefEstimate]:
        """Estimate several candidates without changing their input order."""

        return [
            self.estimate_one(
                candidate=candidate,
                archive=archive,
                top_neighbours=top_neighbours,
                exclude_exact_match=exclude_exact_match,
            )
            for candidate in candidates
        ]

    def kernel_weight(self, similarity_value: float) -> float:
        """Convert raw similarity into a local evidence weight."""

        if not math.isfinite(similarity_value) or not 0.0 <= similarity_value <= 1.0:
            raise ValueError("similarity_value must be finite and inside [0, 1]")

        distance = 1.0 - similarity_value
        denominator = 2.0 * self.kernel_bandwidth**2
        return float(math.exp(-(distance**2) / denominator))

    @staticmethod
    def _archive_prior_mean(
        archive: EvaluatedArchitectureArchive,
        candidate: ArchitectureEncoding,
        exclude_exact_match: bool,
    ) -> float:
        """Return a safe fallback mean when no kernel neighbour is available."""

        values = [
            entry.fitness_mean
            for entry in archive.entries()
            if not (exclude_exact_match and entry.architecture_id == candidate.architecture_id)
        ]
        if not values:
            values = archive.fitness_values()
        return float(sum(values) / len(values))


def _encoding(
    architecture_id: str,
    modules: List[str],
    fitness_length: float,
) -> ArchitectureEncoding:
    """Create a simple encoding for the local self-test."""

    base_map = {
        "ca-densenet": ("densenet", "ca"),
        "cbam-resnet": ("resnet", "cbam"),
        "se-resnet": ("resnet", "se"),
        "pool": ("pool", "none"),
    }
    bases = [base_map[module][0] for module in modules]
    attentions = [base_map[module][1] for module in modules]
    pairs = [f"{attention}-{base}" for attention, base in zip(attentions, bases)]

    from collections import Counter

    return ArchitectureEncoding(
        architecture_id=architecture_id,
        architecture_string="-".join(modules),
        individual_id=architecture_id,
        length=len(modules),
        module_sequence=modules,
        base_sequence=bases,
        attention_sequence=attentions,
        base_attention_pairs=pairs,
        module_counts=dict(Counter(modules)),
        base_counts=dict(Counter(bases)),
        attention_counts=dict(Counter(attentions)),
        module_bigrams=[f"{a}->{b}" for a, b in zip(modules, modules[1:])],
        pair_bigrams=[f"{a}->{b}" for a, b in zip(pairs, pairs[1:])],
        numeric_summary={"length": fitness_length, "attention_density": 0.5},
        unit_records=[],
    )


def _run_self_test() -> None:
    """Check archive-wide weighting and exact-match exclusion."""

    archive = EvaluatedArchitectureArchive()
    good = _encoding("good", ["ca-densenet", "pool", "cbam-resnet"], 3.0)
    lower = _encoding("lower", ["ca-densenet", "pool", "se-resnet"], 3.0)
    candidate = _encoding("candidate", ["ca-densenet", "pool", "cbam-resnet"], 3.0)

    archive.add_encoding(good, fitness=0.84, generation=0)
    archive.add_encoding(lower, fitness=0.80, generation=0)

    estimator = SimilarityBeliefEstimator(kernel_bandwidth=0.25)
    result = estimator.estimate_one(candidate, archive)

    assert 0.80 <= result.belief_mean <= 0.84
    assert result.evidence_strength > 0
    assert result.effective_neighbour_count >= 1.0
    assert result.max_similarity == 1.0
    assert result.used_neighbour_count == 2

    print("Similarity belief estimator self-test passed.")
    print(result.to_dict())


if __name__ == "__main__":
    _run_self_test()
