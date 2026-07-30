"""
This module calculates uncertainty for one pre-evaluation fitness belief.

The raw uncertainty combines archive variance, effective neighbour count, and
local neighbour disagreement. A lightweight calibrator can later map these
signals to expected absolute prediction error using completed warm-up cycles.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from .archive import EvaluatedArchitectureArchive
from .calibration import UncertaintyCalibrator
from .estimator import BeliefEstimate


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Store raw and calibrated uncertainty information."""

    uncertainty: float
    raw_uncertainty: float
    archive_variance: float
    evidence_weakness: float
    local_standard_deviation: float
    neighbour_sparsity: float
    calibrated: bool

    def to_dict(self) -> Dict[str, object]:
        """Return the uncertainty result as a plain dictionary."""

        return asdict(self)


class BeliefUncertaintyEstimator:
    """Calculate candidate uncertainty without training a fitness predictor."""

    def __init__(self, calibrator: Optional[UncertaintyCalibrator] = None) -> None:
        """Create the estimator with an optional learned error calibrator."""

        self.calibrator = calibrator

    def estimate(
        self,
        belief: BeliefEstimate,
        archive: EvaluatedArchitectureArchive,
    ) -> UncertaintyEstimate:
        """Calculate raw and optionally calibrated uncertainty."""

        archive_variance = self._archive_variance(archive)
        
        effective_count = max(
            0.0,
            belief.effective_neighbour_count,
        )
        
        evidence_strength = max(
            0.0,
            belief.evidence_strength,
        )
        
        effective_evidence = min(
            effective_count,
            evidence_strength,
        )
        
        epistemic_variance = (
            archive_variance
            / (1.0 + effective_evidence)
        )
        
        local_variance = max(0.0,
            belief.neighbour_disagreement,
        )

        if belief.model_variance is not None:
            raw_variance = max(0.0, belief.model_variance) + local_variance
        else:
            raw_variance = epistemic_variance + local_variance

        raw_uncertainty = math.sqrt(max(raw_variance, 1e-16))
        evidence_weakness = 1.0 / (1.0 + evidence_strength)
        local_standard_deviation = math.sqrt(local_variance)
        neighbour_sparsity = 1.0 / (1.0 + effective_evidence)

        calibrated = bool(self.calibrator and self.calibrator.state.fitted)
        uncertainty = raw_uncertainty
        if self.calibrator is not None:
            uncertainty = self.calibrator.predict(
                evidence_strength=belief.evidence_strength,
                neighbour_disagreement=belief.neighbour_disagreement,
                effective_neighbour_count=belief.effective_neighbour_count,
                fallback=raw_uncertainty,
            )

        return UncertaintyEstimate(
            uncertainty=float(uncertainty),
            raw_uncertainty=float(raw_uncertainty),
            archive_variance=float(archive_variance),
            evidence_weakness=float(evidence_weakness),
            local_standard_deviation=float(local_standard_deviation),
            neighbour_sparsity=float(neighbour_sparsity),
            calibrated=calibrated,
        )

    @staticmethod
    def _archive_variance(archive: EvaluatedArchitectureArchive) -> float:
        """Return a stable population variance for archive fitness values."""

        values = archive.fitness_values()
        if len(values) < 2:
            return 1e-4
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return float(max(variance, 1e-8))
