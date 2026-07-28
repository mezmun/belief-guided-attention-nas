"""
This module evaluates how well pre-training belief scores match real fitness.

The main target is ranking quality, because the belief method is designed to
prioritize candidate architectures rather than predict exact accuracy. The
module also reports numerical error and optional uncertainty calibration.
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

    def to_dict(self) -> Dict[str, object]:
        """Return the metrics as a plain dictionary."""

        return asdict(self)


class BeliefMetricsCalculator:
    """Calculate cycle-wise ranking, error, and calibration metrics."""

    VERSION = "1.0"

    def calculate_cycle(
        self,
        records: Iterable[EvaluatedBeliefRecord],
        top_k: int = 5,
    ) -> CycleMetrics:
        """Calculate metrics from completed records belonging to one cycle."""

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
        belief_order = sorted(
            range(len(items)), key=lambda index: beliefs[index], reverse=True
        )
        true_order = sorted(
            range(len(items)), key=lambda index: fitness[index], reverse=True
        )
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

        return CycleMetrics(
            run_id=next(iter(run_ids)),
            cycle=next(iter(cycles)),
            evaluated_count=len(items),
            spearman_correlation=self._spearman(beliefs, fitness),
            pearson_correlation=self._pearson(beliefs, fitness),
            mae=float(sum(errors) / len(errors)),
            rmse=float(math.sqrt(sum(squared_errors) / len(squared_errors))),
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
        )

    def calculate_all_cycles(
        self,
        records: Iterable[EvaluatedBeliefRecord],
        top_k: int = 5,
    ) -> List[CycleMetrics]:
        """Group records by run and cycle, then calculate ordered summaries."""

        grouped: Dict[tuple[str, int], List[EvaluatedBeliefRecord]] = {}
        for record in records:
            grouped.setdefault((record.run_id, record.cycle), []).append(record)

        return [
            self.calculate_cycle(grouped[key], top_k=top_k)
            for key in sorted(grouped, key=lambda value: (value[0], value[1]))
        ]

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
        """Return Pearson correlation or None for constant/short inputs."""

        if len(left) != len(right):
            raise ValueError("Correlation inputs must have the same length")
        if len(left) < 2:
            return None

        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left_centered = [value - left_mean for value in left]
        right_centered = [value - right_mean for value in right]
        numerator = sum(a * b for a, b in zip(left_centered, right_centered))
        left_scale = math.sqrt(sum(value**2 for value in left_centered))
        right_scale = math.sqrt(sum(value**2 for value in right_centered))
        denominator = left_scale * right_scale
        if denominator <= 0:
            return None
        return float(numerator / denominator)

    @staticmethod
    def _average_ranks(values: Sequence[float]) -> List[float]:
        """Return ascending average ranks with correct handling of ties."""

        indexed = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        position = 0

        while position < len(indexed):
            end = position + 1
            while end < len(indexed) and indexed[end][1] == indexed[position][1]:
                end += 1

            average_rank = ((position + 1) + end) / 2.0
            for offset in range(position, end):
                original_index = indexed[offset][0]
                ranks[original_index] = average_rank
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


def _record(index: int, belief: float, fitness: float) -> EvaluatedBeliefRecord:
    """Create a compact completed record for the local self-test."""

    error = fitness - belief
    return EvaluatedBeliefRecord(
        record_id=f"test:1:indi{index}:arch{index}",
        run_id="test",
        cycle=1,
        architecture_id=f"arch{index}",
        individual_id=f"indi{index}",
        architecture_string=f"architecture-{index}",
        belief_mean=belief,
        evidence_strength=1.0 + index,
        effective_neighbour_count=2.0 + index,
        neighbour_disagreement=0.001 * index,
        max_similarity=0.7,
        used_neighbour_count=3,
        used_prior_only=False,
        belief_uncertainty=0.01 + index * 0.005,
        novelty=None,
        selected_for_evaluation=True,
        selection_reason="monitor_all",
        true_fitness=fitness,
        absolute_error=abs(error),
        squared_error=error**2,
        evaluation_source="training",
        created_at_utc="2026-01-01T00:00:00+00:00",
        evaluated_at_utc="2026-01-01T01:00:00+00:00",
    )


def _run_self_test() -> None:
    """Check perfect ranking and top-k matching on a simple cycle."""

    records = [
        _record(1, 0.78, 0.79),
        _record(2, 0.81, 0.82),
        _record(3, 0.80, 0.805),
        _record(4, 0.84, 0.835),
    ]
    metrics = BeliefMetricsCalculator().calculate_cycle(records, top_k=2)

    assert metrics.spearman_correlation is not None
    assert abs(metrics.spearman_correlation - 1.0) < 1e-12
    assert metrics.top_k_hit_count == 2
    assert metrics.top_k_recall == 1.0
    assert metrics.evaluated_count == 4

    print("Belief metrics self-test passed.")
    print(metrics.to_dict())


if __name__ == "__main__":
    _run_self_test()
