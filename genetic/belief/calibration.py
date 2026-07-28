"""
This module learns lightweight calibration values from completed evaluations.

It does not train a fitness surrogate. The uncertainty calibrator only learns
how raw belief evidence relates to later prediction error. The similarity
calibrator learns interpretable non-negative weights for existing similarity
components using only previously evaluated architectures.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .archive import EvaluatedArchitectureArchive
from .monitoring import EvaluatedBeliefRecord
from .similarity import ArchitectureSimilarity, SimilarityWeights


@dataclass(frozen=True)
class UncertaintyCalibrationState:
    """Store the fitted uncertainty calibration model."""

    fitted: bool
    sample_count: int
    intercept: float
    coefficients: List[float]
    feature_means: List[float]
    feature_scales: List[float]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-compatible state dictionary."""

        return asdict(self)


class UncertaintyCalibrator:
    """Predict absolute belief error from evidence quality features."""

    FEATURE_COUNT = 3

    def __init__(self, ridge_alpha: float = 1.0) -> None:
        """Create an unfitted ridge calibrator."""

        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be greater than zero")
        self.ridge_alpha = float(ridge_alpha)
        self.state = UncertaintyCalibrationState(
            fitted=False,
            sample_count=0,
            intercept=0.0,
            coefficients=[0.0] * self.FEATURE_COUNT,
            feature_means=[0.0] * self.FEATURE_COUNT,
            feature_scales=[1.0] * self.FEATURE_COUNT,
        )

    @staticmethod
    def feature_vector(
        evidence_strength: float,
        neighbour_disagreement: float,
        effective_neighbour_count: float,
    ) -> List[float]:
        """Build stable uncertainty features from one belief estimate."""

        evidence_weakness = 1.0 / (1.0 + max(0.0, float(evidence_strength)))
        local_standard_deviation = math.sqrt(max(0.0, float(neighbour_disagreement)))
        neighbour_sparsity = 1.0 / (1.0 + max(0.0, float(effective_neighbour_count)))
        return [evidence_weakness, local_standard_deviation, neighbour_sparsity]

    def fit(self, records: Iterable[EvaluatedBeliefRecord]) -> bool:
        """Fit the calibration model from completed evaluation records."""

        samples = [
            {
                "evidence_strength": item.evidence_strength,
                "neighbour_disagreement": item.neighbour_disagreement,
                "effective_neighbour_count": item.effective_neighbour_count,
                "absolute_error": item.absolute_error,
            }
            for item in records
        ]
        return self.fit_samples(samples)

    def fit_samples(self, samples: Iterable[Dict[str, float]]) -> bool:
        """Fit from compact samples that can be saved in the belief state."""

        items = list(samples)
        if len(items) < 2:
            return False

        x = np.asarray(
            [
                self.feature_vector(
                    item["evidence_strength"],
                    item["neighbour_disagreement"],
                    item["effective_neighbour_count"],
                )
                for item in items
            ],
            dtype=np.float64,
        )
        y = np.asarray([item["absolute_error"] for item in items], dtype=np.float64)

        means = x.mean(axis=0)
        scales = x.std(axis=0)
        scales[scales < 1e-12] = 1.0
        standardized = (x - means) / scales
        design = np.column_stack([np.ones(len(items)), standardized])

        penalty = np.eye(design.shape[1], dtype=np.float64) * self.ridge_alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)

        self.state = UncertaintyCalibrationState(
            fitted=True,
            sample_count=len(items),
            intercept=float(max(0.0, coefficients[0])),
            coefficients=[float(value) for value in coefficients[1:]],
            feature_means=[float(value) for value in means],
            feature_scales=[float(value) for value in scales],
        )
        return True

    def predict(
        self,
        evidence_strength: float,
        neighbour_disagreement: float,
        effective_neighbour_count: float,
        fallback: float,
    ) -> float:
        """Return calibrated expected error or a safe fallback value."""

        if not self.state.fitted:
            return max(0.0, float(fallback))

        features = np.asarray(
            self.feature_vector(
                evidence_strength,
                neighbour_disagreement,
                effective_neighbour_count,
            ),
            dtype=np.float64,
        )
        means = np.asarray(self.state.feature_means, dtype=np.float64)
        scales = np.asarray(self.state.feature_scales, dtype=np.float64)
        coefficients = np.asarray(self.state.coefficients, dtype=np.float64)
        standardized = (features - means) / scales
        prediction = self.state.intercept + float(standardized @ coefficients)
        return float(max(1e-8, prediction))

    def load_state(self, data: Dict[str, object]) -> None:
        """Restore a previously saved calibration state."""

        state = UncertaintyCalibrationState(
            fitted=bool(data.get("fitted", False)),
            sample_count=int(data.get("sample_count", 0)),
            intercept=float(data.get("intercept", 0.0)),
            coefficients=[float(value) for value in data.get("coefficients", [])],
            feature_means=[float(value) for value in data.get("feature_means", [])],
            feature_scales=[float(value) for value in data.get("feature_scales", [])],
        )
        if len(state.coefficients) != self.FEATURE_COUNT:
            raise ValueError("Invalid uncertainty coefficient count")
        if len(state.feature_means) != self.FEATURE_COUNT:
            raise ValueError("Invalid uncertainty feature mean count")
        if len(state.feature_scales) != self.FEATURE_COUNT:
            raise ValueError("Invalid uncertainty feature scale count")
        self.state = state


class SimilarityWeightCalibrator:
    """Learn non-negative weights for interpretable similarity components."""

    COMPONENT_NAMES = [
        "module_sequence",
        "base_sequence",
        "attention_sequence",
        "count_similarity",
        "module_bigram",
        "pair_bigram",
        "numeric_similarity",
    ]

    def __init__(
        self,
        target_tau: float = 0.02,
        ridge_alpha: float = 1.0,
        max_pairs: int = 5000,
        random_seed: int = 2312390,
    ) -> None:
        """Create a deterministic pairwise similarity calibrator."""

        if target_tau <= 0 or ridge_alpha <= 0 or max_pairs < 1:
            raise ValueError("Similarity calibration settings must be positive")
        self.target_tau = float(target_tau)
        self.ridge_alpha = float(ridge_alpha)
        self.max_pairs = int(max_pairs)
        self.random = random.Random(int(random_seed))
        self.last_pair_count = 0

    def fit(
        self,
        archive: EvaluatedArchitectureArchive,
        calculator: Optional[ArchitectureSimilarity] = None,
    ) -> Optional[SimilarityWeights]:
        """Fit weights from past archive pairs and return None when insufficient."""

        entries = archive.entries()
        pair_indices = [
            (left, right)
            for left in range(len(entries))
            for right in range(left + 1, len(entries))
        ]
        if not pair_indices:
            return None
        if len(pair_indices) > self.max_pairs:
            pair_indices = self.random.sample(pair_indices, self.max_pairs)

        similarity = calculator or ArchitectureSimilarity()
        rows: List[List[float]] = []
        targets: List[float] = []
        for left_index, right_index in pair_indices:
            left = entries[left_index]
            right = entries[right_index]
            breakdown = similarity.compare(left.encoding, right.encoding)
            rows.append([float(getattr(breakdown, name)) for name in self.COMPONENT_NAMES])
            difference = abs(left.fitness_mean - right.fitness_mean)
            targets.append(math.exp(-difference / self.target_tau))

        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        penalty = np.eye(x.shape[1], dtype=np.float64) * self.ridge_alpha
        coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        coefficients = np.clip(coefficients, 0.0, None)
        total = float(coefficients.sum())
        if total <= 1e-12:
            return None

        normalized = coefficients / total
        self.last_pair_count = len(pair_indices)
        return SimilarityWeights(
            **{
                name: float(value)
                for name, value in zip(self.COMPONENT_NAMES, normalized)
            }
        )
