"""
This module calculates interpretable similarity between two architectures.

The similarity score combines module order, backbone order, attention order,
component counts, local transitions, and numeric architecture properties. Each
component remains available for analysis instead of being hidden in one model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, Iterable, List, Mapping, Sequence

from .encoder import ArchitectureEncoding


@dataclass(frozen=True)
class SimilarityWeights:
    """Store the weights used to combine similarity components."""

    module_sequence: float = 0.25
    base_sequence: float = 0.15
    attention_sequence: float = 0.15
    count_similarity: float = 0.20
    module_bigram: float = 0.10
    pair_bigram: float = 0.10
    numeric_similarity: float = 0.05

    def normalized(self) -> "SimilarityWeights":
        """Return non-negative weights scaled to a total of one."""

        values = asdict(self)
        if any(not isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Similarity weights must be finite and non-negative")

        total = sum(values.values())
        if total <= 0:
            raise ValueError("At least one similarity weight must be positive")

        return SimilarityWeights(**{key: value / total for key, value in values.items()})


@dataclass(frozen=True)
class SimilarityBreakdown:
    """Store the final similarity and all interpretable component scores."""

    total: float
    module_sequence: float
    base_sequence: float
    attention_sequence: float
    count_similarity: float
    module_bigram: float
    pair_bigram: float
    numeric_similarity: float

    def to_dict(self) -> Dict[str, float]:
        """Return the similarity result as a plain dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class SimilarityMatrix:
    """Store a candidate-by-reference similarity matrix."""

    candidate_ids: List[str]
    reference_ids: List[str]
    values: List[List[float]]

    @property
    def shape(self) -> tuple[int, int]:
        """Return matrix dimensions as (candidate_count, reference_count)."""

        return len(self.candidate_ids), len(self.reference_ids)


class ArchitectureSimilarity:
    """Calculate detailed similarity between deterministic encodings."""

    VERSION = "1.0"

    def __init__(self, weights: SimilarityWeights | None = None) -> None:
        """Create the calculator with normalized component weights."""

        self.weights = (weights or SimilarityWeights()).normalized()

    def compare(
        self,
        left: ArchitectureEncoding,
        right: ArchitectureEncoding,
    ) -> SimilarityBreakdown:
        """Compare two architecture encodings and return all components."""

        module_sequence = self._sequence_similarity(
            left.module_sequence, right.module_sequence
        )
        base_sequence = self._sequence_similarity(left.base_sequence, right.base_sequence)
        attention_sequence = self._sequence_similarity(
            left.attention_sequence, right.attention_sequence
        )

        count_similarity = self._mean(
            [
                self._counter_similarity(left.module_counts, right.module_counts),
                self._counter_similarity(left.base_counts, right.base_counts),
                self._counter_similarity(left.attention_counts, right.attention_counts),
                self._counter_similarity(
                    Counter(left.base_attention_pairs),
                    Counter(right.base_attention_pairs),
                ),
            ]
        )

        module_bigram = self._counter_similarity(
            Counter(left.module_bigrams), Counter(right.module_bigrams)
        )
        pair_bigram = self._counter_similarity(
            Counter(left.pair_bigrams), Counter(right.pair_bigrams)
        )
        numeric_similarity = self._numeric_similarity(
            left.numeric_summary, right.numeric_summary
        )

        weighted_total = (
            self.weights.module_sequence * module_sequence
            + self.weights.base_sequence * base_sequence
            + self.weights.attention_sequence * attention_sequence
            + self.weights.count_similarity * count_similarity
            + self.weights.module_bigram * module_bigram
            + self.weights.pair_bigram * pair_bigram
            + self.weights.numeric_similarity * numeric_similarity
        )

        return SimilarityBreakdown(
            total=self._clip(weighted_total),
            module_sequence=module_sequence,
            base_sequence=base_sequence,
            attention_sequence=attention_sequence,
            count_similarity=count_similarity,
            module_bigram=module_bigram,
            pair_bigram=pair_bigram,
            numeric_similarity=numeric_similarity,
        )

    def build_matrix(
        self,
        candidates: Iterable[ArchitectureEncoding],
        references: Iterable[ArchitectureEncoding],
    ) -> SimilarityMatrix:
        """Build a candidate-by-reference matrix without changing input order."""

        candidate_list = list(candidates)
        reference_list = list(references)
        values = [
            [self.compare(candidate, reference).total for reference in reference_list]
            for candidate in candidate_list
        ]

        return SimilarityMatrix(
            candidate_ids=[item.architecture_id for item in candidate_list],
            reference_ids=[item.architecture_id for item in reference_list],
            values=values,
        )

    @classmethod
    def _sequence_similarity(cls, left: Sequence[str], right: Sequence[str]) -> float:
        """Use normalized longest common subsequence for variable-length order."""

        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        previous = [0] * (len(right) + 1)
        for left_value in left:
            current = [0]
            for index, right_value in enumerate(right, start=1):
                if left_value == right_value:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current

        return cls._clip(previous[-1] / max(len(left), len(right)))

    @classmethod
    def _counter_similarity(
        cls,
        left: Mapping[str, int],
        right: Mapping[str, int],
    ) -> float:
        """Use generalized Jaccard similarity for count-based features."""

        keys = set(left).union(right)
        if not keys:
            return 1.0

        intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
        union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
        if union == 0:
            return 1.0
        return cls._clip(intersection / union)

    @classmethod
    def _numeric_similarity(
        cls,
        left: Mapping[str, float],
        right: Mapping[str, float],
    ) -> float:
        """Average scale-free similarity across shared numeric features."""

        keys = set(left).intersection(right)
        if not keys:
            return 1.0

        similarities: List[float] = []
        for key in sorted(keys):
            left_value = float(left[key])
            right_value = float(right[key])
            scale = max(abs(left_value), abs(right_value), 1e-12)
            similarity = 1.0 - abs(left_value - right_value) / scale
            similarities.append(cls._clip(similarity))

        return cls._mean(similarities)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Return the arithmetic mean, or one for two empty feature groups."""

        if not values:
            return 1.0
        return float(sum(values) / len(values))

    @staticmethod
    def _clip(value: float) -> float:
        """Keep a floating-point similarity value inside the closed unit range."""

        return float(min(1.0, max(0.0, value)))


def _make_encoding(
    architecture_id: str,
    modules: List[str],
    bases: List[str],
    attentions: List[str],
) -> ArchitectureEncoding:
    """Create a small encoding used by the local self-test."""

    pairs = [f"{attention}-{base}" for attention, base in zip(attentions, bases)]
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
        numeric_summary={"length": float(len(modules)), "attention_density": 0.5},
        unit_records=[],
    )


def _run_self_test() -> None:
    """Check identity, symmetry, range, and matrix orientation."""

    first = _make_encoding(
        "a", ["ca-densenet", "pool", "cbam-resnet"],
        ["densenet", "pool", "resnet"], ["ca", "none", "cbam"]
    )
    second = _make_encoding(
        "b", ["ca-densenet", "pool", "se-resnet"],
        ["densenet", "pool", "resnet"], ["ca", "none", "se"]
    )

    calculator = ArchitectureSimilarity()
    identity = calculator.compare(first, first)
    forward = calculator.compare(first, second)
    backward = calculator.compare(second, first)
    matrix = calculator.build_matrix([first, second], [first])

    assert abs(identity.total - 1.0) < 1e-12
    assert 0.0 <= forward.total <= 1.0
    assert abs(forward.total - backward.total) < 1e-12
    assert matrix.shape == (2, 1)
    assert matrix.values[0][0] == 1.0

    print("Architecture similarity self-test passed.")
    print(forward.to_dict())


if __name__ == "__main__":
    _run_self_test()
