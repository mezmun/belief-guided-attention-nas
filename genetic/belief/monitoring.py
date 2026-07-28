"""
This module records belief values before and after real model evaluation.

Pre-evaluation rows are written before training starts. Real fitness is attached
later to the same record, which prevents accidental fitness leakage and makes
cycle-wise validation possible.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .encoder import ArchitectureEncoding
from .estimator import BeliefEstimate


@dataclass(frozen=True)
class PreEvaluationRecord:
    """Store all information known before real evaluation."""

    record_id: str
    run_id: str
    cycle: int
    architecture_id: str
    individual_id: str
    architecture_string: str
    belief_mean: float
    evidence_strength: float
    effective_neighbour_count: float
    neighbour_disagreement: float
    max_similarity: float
    used_neighbour_count: int
    used_prior_only: bool
    model_variance: Optional[float]
    belief_uncertainty: Optional[float]
    raw_uncertainty: Optional[float]
    novelty: Optional[float]
    selected_for_evaluation: bool
    selection_reason: str
    selection_score: Optional[float]
    created_at_utc: str

    def to_dict(self) -> Dict[str, object]:
        """Return the record as a plain dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluatedBeliefRecord:
    """Combine one pre-evaluation record with its later true fitness."""

    record_id: str
    run_id: str
    cycle: int
    architecture_id: str
    individual_id: str
    architecture_string: str
    belief_mean: float
    evidence_strength: float
    effective_neighbour_count: float
    neighbour_disagreement: float
    max_similarity: float
    used_neighbour_count: int
    used_prior_only: bool
    model_variance: Optional[float]
    belief_uncertainty: Optional[float]
    raw_uncertainty: Optional[float]
    novelty: Optional[float]
    selected_for_evaluation: bool
    selection_reason: str
    selection_score: Optional[float]
    true_fitness: float
    absolute_error: float
    squared_error: float
    evaluation_source: str
    created_at_utc: str
    evaluated_at_utc: str

    def to_dict(self) -> Dict[str, object]:
        """Return the record as a plain dictionary."""

        return asdict(self)


class BeliefCycleMonitor:
    """Track pre-evaluation beliefs and their later real fitness values."""

    VERSION = "2.0"

    def __init__(
        self,
        pre_evaluation_csv: Optional[Path] = None,
        evaluated_csv: Optional[Path] = None,
    ) -> None:
        """Create an empty monitor with optional append-only CSV files."""

        self.pre_evaluation_csv = Path(pre_evaluation_csv) if pre_evaluation_csv else None
        self.evaluated_csv = Path(evaluated_csv) if evaluated_csv else None
        self._pre_records: Dict[str, PreEvaluationRecord] = {}
        self._evaluated_records: Dict[str, EvaluatedBeliefRecord] = {}

    def register_pre_evaluation(
        self,
        encoding: ArchitectureEncoding,
        estimate: BeliefEstimate,
        run_id: str,
        cycle: int,
        selected_for_evaluation: bool,
        selection_reason: str,
        belief_uncertainty: Optional[float] = None,
        raw_uncertainty: Optional[float] = None,
        novelty: Optional[float] = None,
        selection_score: Optional[float] = None,
    ) -> PreEvaluationRecord:
        """Record one candidate before true fitness becomes available."""

        if encoding.architecture_id != estimate.architecture_id:
            raise ValueError("Encoding and belief estimate refer to different architectures")
        record_id = self.make_record_id(
            run_id=run_id,
            cycle=cycle,
            individual_id=encoding.individual_id,
            architecture_id=encoding.architecture_id,
        )
        if record_id in self._pre_records:
            raise ValueError(f"A pre-evaluation record already exists: {record_id}")

        record = PreEvaluationRecord(
            record_id=record_id,
            run_id=str(run_id),
            cycle=int(cycle),
            architecture_id=encoding.architecture_id,
            individual_id=encoding.individual_id,
            architecture_string=encoding.architecture_string,
            belief_mean=self._finite(estimate.belief_mean),
            evidence_strength=self._non_negative(estimate.evidence_strength),
            effective_neighbour_count=self._non_negative(
                estimate.effective_neighbour_count
            ),
            neighbour_disagreement=self._non_negative(
                estimate.neighbour_disagreement
            ),
            max_similarity=self._unit_interval(estimate.max_similarity),
            used_neighbour_count=int(estimate.used_neighbour_count),
            used_prior_only=bool(estimate.used_prior_only),
            model_variance=self._optional_non_negative(estimate.model_variance),
            belief_uncertainty=self._optional_non_negative(belief_uncertainty),
            raw_uncertainty=self._optional_non_negative(raw_uncertainty),
            novelty=self._optional_unit_interval(novelty),
            selected_for_evaluation=bool(selected_for_evaluation),
            selection_reason=str(selection_reason),
            selection_score=self._optional_finite(selection_score),
            created_at_utc=self._utc_now(),
        )
        self._pre_records[record_id] = record
        if self.pre_evaluation_csv is not None:
            self.append_csv(self.pre_evaluation_csv, record.to_dict())
        return record

    def register_evaluation(
        self,
        record_id: str,
        true_fitness: float,
        evaluation_source: str = "training",
    ) -> EvaluatedBeliefRecord:
        """Attach true fitness to one selected pre-evaluation record."""

        if record_id not in self._pre_records:
            raise KeyError(f"Pre-evaluation record was not found: {record_id}")
        if record_id in self._evaluated_records:
            raise ValueError(f"Real fitness was already attached to record: {record_id}")

        pre = self._pre_records[record_id]
        if not pre.selected_for_evaluation:
            raise ValueError("Cannot attach fitness to a candidate marked as not selected")

        fitness = self._finite(true_fitness)
        error = fitness - pre.belief_mean
        record = EvaluatedBeliefRecord(
            **pre.to_dict(),
            true_fitness=fitness,
            absolute_error=abs(error),
            squared_error=error**2,
            evaluation_source=str(evaluation_source),
            evaluated_at_utc=self._utc_now(),
        )
        self._evaluated_records[record_id] = record
        if self.evaluated_csv is not None:
            self.append_csv(self.evaluated_csv, record.to_dict())
        return record

    def pre_records(self, cycle: Optional[int] = None) -> List[PreEvaluationRecord]:
        """Return all pre-evaluation records, optionally for one cycle."""

        records = list(self._pre_records.values())
        return records if cycle is None else [item for item in records if item.cycle == cycle]

    def evaluated_records(
        self, cycle: Optional[int] = None
    ) -> List[EvaluatedBeliefRecord]:
        """Return completed records, optionally for one cycle."""

        records = list(self._evaluated_records.values())
        return records if cycle is None else [item for item in records if item.cycle == cycle]

    @staticmethod
    def make_record_id(
        run_id: str,
        cycle: int,
        individual_id: str,
        architecture_id: str,
    ) -> str:
        """Build a stable identifier for one candidate inside one cycle."""

        return f"{run_id}:{int(cycle)}:{individual_id}:{architecture_id}"

    @staticmethod
    def append_csv(path: Path, row: Dict[str, object]) -> None:
        """Append one row and create the header when needed."""

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _utc_now() -> str:
        """Return an ISO timestamp in UTC."""

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _finite(value: float) -> float:
        clean = float(value)
        if not math.isfinite(clean):
            raise ValueError("A recorded numeric value must be finite")
        return clean

    @classmethod
    def _non_negative(cls, value: float) -> float:
        clean = cls._finite(value)
        if clean < 0:
            raise ValueError("A recorded value must be non-negative")
        return clean

    @classmethod
    def _unit_interval(cls, value: float) -> float:
        clean = cls._finite(value)
        if not 0.0 <= clean <= 1.0:
            raise ValueError("A recorded value must be inside [0, 1]")
        return clean

    @classmethod
    def _optional_non_negative(cls, value: Optional[float]) -> Optional[float]:
        return None if value is None else cls._non_negative(value)

    @classmethod
    def _optional_unit_interval(cls, value: Optional[float]) -> Optional[float]:
        return None if value is None else cls._unit_interval(value)

    @classmethod
    def _optional_finite(cls, value: Optional[float]) -> Optional[float]:
        return None if value is None else cls._finite(value)
