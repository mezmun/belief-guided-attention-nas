"""
This module evaluates how pre-training belief scores match real fitness.

Ranking quality is the main target because the method prioritizes candidates
instead of predicting exact accuracy. Random audit rows are also summarized to
reduce the effect of selection bias during guided search.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .monitoring import EvaluatedBeliefRecord


@dataclass(frozen=True)
class CycleMetrics:
    """Store validation metrics for one evaluated offspring cycle."""

    run_id: str
    cycle: int
    evaluated_count: int
    spearman_correlation: Optional[float]
    pearson_correlation: Optional[float]
    mae: float
    rmse: float
    top_k: int
    top_k_hit_count: int
    top_k_recall: float
    belief_top_k_mean_fitness: float
    true_top_k_mean_fitness: float
    overall_mean_fitness: float
    best_true_fitness: float
    uncertainty_error_correlation: Optional[float]
    random_audit_count: int
    random_audit_spearman: Optional[float]
    random_audit_mae: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        """Return the metrics as a plain dictionary."""

        return asdict(self)


class BeliefMetricsCalculator:
    """Calculate cycle-wise ranking, error, and calibration metrics."""

    VERSION = "2.0"

    def calculate_cycle(
        self,
        records: Iterable[EvaluatedBeliefRecord],
        top_k: int = 5,
    ) -> CycleMetrics:
        """Calculate metrics from completed rows belonging to one cycle."""

        items = list(records)
        if not items:
            raise ValueError("At least one evaluated belief record is required")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        run_ids = {item.run_id for item in items}
        cycles = {item.cycle for item in items}
        if len(run_ids) != 1 or len(cycles) != 1:
            raise ValueError("All records must belong to the same run and cycle")

        beliefs = [item.belief_mean for item in items]
        fitness = [item.true_fitness for item in items]
        errors = [item.absolute_error for item in items]
        squared_errors = [item.squared_error for item in items]

        effective_k = min(top_k, len(items))
        belief_order = sorted(range(len(items)), key=lambda index: beliefs[index], reverse=True)
        true_order = sorted(range(len(items)), key=lambda index: fitness[index], reverse=True)
        belief_top = set(belief_order[:effective_k])
        true_top = set(true_order[:effective_k])
        hit_count = len(belief_top.intersection(true_top))

        uncertainty_pairs = [
            (item.belief_uncertainty, item.absolute_error)
            for item in items
            if item.belief_uncertainty is not None
        ]
        uncertainty_error_correlation = None
        if len(uncertainty_pairs) >= 2:
            uncertainty_error_correlation = self._pearson(
                [float(pair[0]) for pair in uncertainty_pairs],
                [pair[1] for pair in uncertainty_pairs],
            )

        audit = [item for item in items if item.selection_reason == "random_audit"]
        audit_spearman = None
        audit_mae = None
        if audit:
            audit_mae = self._mean([item.absolute_error for item in audit])
        if len(audit) >= 2:
            audit_spearman = self._spearman(
                [item.belief_mean for item in audit],
                [item.true_fitness for item in audit],
            )

        return CycleMetrics(
            run_id=next(iter(run_ids)),
            cycle=next(iter(cycles)),
            evaluated_count=len(items),
            spearman_correlation=self._spearman(beliefs, fitness),
            pearson_correlation=self._pearson(beliefs, fitness),
            mae=self._mean(errors),
            rmse=float(math.sqrt(self._mean(squared_errors))),
            top_k=effective_k,
            top_k_hit_count=hit_count,
            top_k_recall=float(hit_count / effective_k),
            belief_top_k_mean_fitness=self._mean(
                [fitness[index] for index in belief_order[:effective_k]]
            ),
            true_top_k_mean_fitness=self._mean(
                [fitness[index] for index in true_order[:effective_k]]
            ),
            overall_mean_fitness=self._mean(fitness),
            best_true_fitness=max(fitness),
            uncertainty_error_correlation=uncertainty_error_correlation,
            random_audit_count=len(audit),
            random_audit_spearman=audit_spearman,
            random_audit_mae=audit_mae,
        )

    @classmethod
    def _spearman(
        cls,
        left: Sequence[float],
        right: Sequence[float],
    ) -> Optional[float]:
        """Return tie-aware Spearman correlation or None when undefined."""

        if len(left) != len(right):
            raise ValueError("Correlation inputs must have the same length")
        if len(left) < 2:
            return None
        return cls._pearson(cls._average_ranks(left), cls._average_ranks(right))

    @staticmethod
    def _pearson(
        left: Sequence[float],
        right: Sequence[float],
    ) -> Optional[float]:
        """Return Pearson correlation or None for constant inputs."""

        if len(left) != len(right):
            raise ValueError("Correlation inputs must have the same length")
        if len(left) < 2:
            return None
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left_centered = [value - left_mean for value in left]
        right_centered = [value - right_mean for value in right]
        denominator = math.sqrt(sum(value**2 for value in left_centered)) * math.sqrt(
            sum(value**2 for value in right_centered)
        )
        if denominator <= 0:
            return None
        return float(
            sum(a * b for a, b in zip(left_centered, right_centered)) / denominator
        )

    @staticmethod
    def _average_ranks(values: Sequence[float]) -> List[float]:
        """Return ascending average ranks with correct tie handling."""

        indexed = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(indexed):
            end = position + 1
            while end < len(indexed) and indexed[end][1] == indexed[position][1]:
                end += 1
            average_rank = ((position + 1) + end) / 2.0
            for offset in range(position, end):
                ranks[indexed[offset][0]] = average_rank
            position = end
        return ranks

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Return a finite arithmetic mean."""

        if not values:
            raise ValueError("Cannot calculate the mean of an empty sequence")
        result = float(sum(values) / len(values))
        if not math.isfinite(result):
            raise ValueError("Metric calculation produced a non-finite mean")
        return result
