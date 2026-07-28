"""
This module propagates known fitness values through architecture similarity.

It supports a direct kernel-weighted belief mean and an optional Gaussian
precision update. Both methods use the full evaluated archive rather than one
reference architecture.
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
    model_variance: Optional[float]
    neighbours: List[NeighbourEvidence]

    def to_dict(self) -> Dict[str, object]:
        """Return the complete estimate as a serializable dictionary."""

        data = asdict(self)
        data["neighbours"] = [item.to_dict() for item in self.neighbours]
        return data


class SimilarityBeliefEstimator:
    """Calculate archive-wide kernel or Bayesian precision fitness beliefs."""

    VERSION = "2.0"

    def __init__(
        self,
        similarity: Optional[ArchitectureSimilarity] = None,
        kernel_bandwidth: float = 0.25,
        minimum_kernel_weight: float = 1e-12,
        method: str = "kernel_mean",
    ) -> None:
        """Create the estimator and validate its settings."""

        if kernel_bandwidth <= 0 or not math.isfinite(kernel_bandwidth):
            raise ValueError("kernel_bandwidth must be finite and greater than zero")
        if minimum_kernel_weight < 0 or not math.isfinite(minimum_kernel_weight):
            raise ValueError("minimum_kernel_weight must be finite and non-negative")
        if method not in {"kernel_mean", "bayesian_precision"}:
            raise ValueError("method must be 'kernel_mean' or 'bayesian_precision'")

        self.similarity = similarity or ArchitectureSimilarity()
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.minimum_kernel_weight = float(minimum_kernel_weight)
        self.method = method

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

        prior_mean, prior_variance = self._archive_prior(
            archive, candidate, exclude_exact_match
        )
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
                model_variance=prior_variance,
                neighbours=[],
            )

        total_weight = sum(item[2] for item in weighted_items)
        kernel_mean = sum(
            entry.fitness_mean * kernel_weight
            for entry, _, kernel_weight in weighted_items
        ) / total_weight
        disagreement = sum(
            kernel_weight * (entry.fitness_mean - kernel_mean) ** 2
            for entry, _, kernel_weight in weighted_items
        ) / total_weight
        squared_weight_sum = sum(item[2] ** 2 for item in weighted_items)
        effective_count = total_weight**2 / squared_weight_sum if squared_weight_sum > 0 else 0.0

        belief_mean = kernel_mean
        model_variance: Optional[float] = None
        if self.method == "bayesian_precision":
            belief_mean, model_variance = self._bayesian_precision_update(
                weighted_items=weighted_items,
                total_weight=total_weight,
                effective_count=effective_count,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
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
            model_variance=model_variance,
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
        return float(
            math.exp(-(distance**2) / (2.0 * self.kernel_bandwidth**2))
        )

    @staticmethod
    def _bayesian_precision_update(
        weighted_items: List[tuple[ArchiveEntry, float, float]],
        total_weight: float,
        effective_count: float,
        prior_mean: float,
        prior_variance: float,
    ) -> tuple[float, float]:
        """Apply a batch Gaussian update equivalent to Kalman precision fusion."""

        safe_prior_variance = max(prior_variance, 1e-8)
        observation_variance = safe_prior_variance
        prior_precision = 1.0 / safe_prior_variance

        weighted_precision_mean = 0.0
        for entry, _, kernel_weight in weighted_items:
            normalized_weight = kernel_weight / total_weight
            effective_weight = normalized_weight * max(effective_count, 1.0)
            weighted_precision_mean += (
                effective_weight * entry.fitness_mean / observation_variance
            )

        observation_precision = max(effective_count, 1.0) / observation_variance
        posterior_precision = prior_precision + observation_precision
        posterior_mean = (
            prior_precision * prior_mean + weighted_precision_mean
        ) / posterior_precision
        posterior_variance = 1.0 / posterior_precision
        return float(posterior_mean), float(posterior_variance)

    @staticmethod
    def _archive_prior(
        archive: EvaluatedArchitectureArchive,
        candidate: ArchitectureEncoding,
        exclude_exact_match: bool,
    ) -> tuple[float, float]:
        """Return archive mean and variance for a safe prior distribution."""

        values = [
            entry.fitness_mean
            for entry in archive.entries()
            if not (exclude_exact_match and entry.architecture_id == candidate.architecture_id)
        ]
        if not values:
            values = archive.fitness_values()
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return float(mean_value), float(max(variance, 1e-8))
