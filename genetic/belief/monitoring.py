"""
This module records belief values before and after real model evaluation.

The pre-evaluation record is written before training starts. After training,
the real fitness is attached to the same record. This separation helps prevent
fitness leakage and supports cycle-wise validation of the belief method.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .encoder import ArchitectureEncoding
from .estimator import BeliefEstimate


@dataclass(frozen=True)
class PreEvaluationRecord:
    """Store all information that is known before real model training."""

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
    belief_uncertainty: Optional[float]
    novelty: Optional[float]
    selected_for_evaluation: bool
    selection_reason: str
    created_at_utc: str

    def to_dict(self) -> Dict[str, object]:
        """Return the record as a plain dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluatedBeliefRecord:
    """Combine one pre-evaluation belief with its later real fitness."""

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
    belief_uncertainty: Optional[float]
    novelty: Optional[float]
    selected_for_evaluation: bool
    selection_reason: str
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

    VERSION = "1.0"

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
        selected_for_evaluation: bool = True,
        selection_reason: str = "monitor_all",
        belief_uncertainty: Optional[float] = None,
        novelty: Optional[float] = None,
    ) -> PreEvaluationRecord:
        """Record one candidate before its true fitness becomes available."""

        if encoding.architecture_id != estimate.architecture_id:
            raise ValueError("Encoding and belief estimate refer to different architectures")
        if cycle < 0:
            raise ValueError("cycle must be zero or greater")

        clean_run_id = str(run_id).strip() or "default_run"
        clean_reason = str(selection_reason).strip() or "unspecified"
        clean_uncertainty = self._optional_non_negative(
            belief_uncertainty, "belief_uncertainty"
        )
        clean_novelty = self._optional_unit_interval(novelty, "novelty")

        record_id = self._make_record_id(
            run_id=clean_run_id,
            cycle=cycle,
            individual_id=encoding.individual_id,
            architecture_id=encoding.architecture_id,
        )
        if record_id in self._pre_records:
            raise ValueError(f"A pre-evaluation record already exists: {record_id}")

        record = PreEvaluationRecord(
            record_id=record_id,
            run_id=clean_run_id,
            cycle=int(cycle),
            architecture_id=encoding.architecture_id,
            individual_id=encoding.individual_id,
            architecture_string=encoding.architecture_string,
            belief_mean=self._finite_float(estimate.belief_mean, "belief_mean"),
            evidence_strength=self._non_negative_float(
                estimate.evidence_strength, "evidence_strength"
            ),
            effective_neighbour_count=self._non_negative_float(
                estimate.effective_neighbour_count, "effective_neighbour_count"
            ),
            neighbour_disagreement=self._non_negative_float(
                estimate.neighbour_disagreement, "neighbour_disagreement"
            ),
            max_similarity=self._unit_interval(
                estimate.max_similarity, "max_similarity"
            ),
            used_neighbour_count=int(estimate.used_neighbour_count),
            used_prior_only=bool(estimate.used_prior_only),
            belief_uncertainty=clean_uncertainty,
            novelty=clean_novelty,
            selected_for_evaluation=bool(selected_for_evaluation),
            selection_reason=clean_reason,
            created_at_utc=self._utc_now(),
        )

        self._pre_records[record_id] = record
        if self.pre_evaluation_csv is not None:
            self._append_csv(self.pre_evaluation_csv, record.to_dict())
        return record

    def register_many_pre_evaluation(
        self,
        encodings: Sequence[ArchitectureEncoding],
        estimates: Sequence[BeliefEstimate],
        run_id: str,
        cycle: int,
        selected_flags: Optional[Sequence[bool]] = None,
        selection_reasons: Optional[Sequence[str]] = None,
    ) -> List[PreEvaluationRecord]:
        """Record several candidates while preserving their current order."""

        if len(encodings) != len(estimates):
            raise ValueError("encodings and estimates must have the same length")

        count = len(encodings)
        flags = list(selected_flags) if selected_flags is not None else [True] * count
        reasons = (
            list(selection_reasons)
            if selection_reasons is not None
            else ["monitor_all"] * count
        )
        if len(flags) != count or len(reasons) != count:
            raise ValueError("Selection metadata must match the candidate count")

        return [
            self.register_pre_evaluation(
                encoding=encoding,
                estimate=estimate,
                run_id=run_id,
                cycle=cycle,
                selected_for_evaluation=flag,
                selection_reason=reason,
            )
            for encoding, estimate, flag, reason in zip(
                encodings, estimates, flags, reasons
            )
        ]

    def register_evaluation(
        self,
        record_id: str,
        true_fitness: float,
        evaluation_source: str = "training",
    ) -> EvaluatedBeliefRecord:
        """Attach real fitness to an existing pre-evaluation record."""

        key = str(record_id)
        if key not in self._pre_records:
            raise KeyError(f"Pre-evaluation record was not found: {key}")
        if key in self._evaluated_records:
            raise ValueError(f"Real fitness was already attached to record: {key}")

        pre = self._pre_records[key]
        if not pre.selected_for_evaluation:
            raise ValueError("Cannot attach fitness to a candidate marked as not selected")

        fitness = self._finite_float(true_fitness, "true_fitness")
        error = fitness - pre.belief_mean
        record = EvaluatedBeliefRecord(
            **pre.to_dict(),
            true_fitness=fitness,
            absolute_error=abs(error),
            squared_error=error**2,
            evaluation_source=str(evaluation_source).strip() or "unknown",
            evaluated_at_utc=self._utc_now(),
        )

        self._evaluated_records[key] = record
        if self.evaluated_csv is not None:
            self._append_csv(self.evaluated_csv, record.to_dict())
        return record

    def register_many_evaluations(
        self,
        record_ids: Sequence[str],
        true_fitness_values: Sequence[float],
        evaluation_source: str = "training",
    ) -> List[EvaluatedBeliefRecord]:
        """Attach real fitness values to several pre-evaluation records."""

        if len(record_ids) != len(true_fitness_values):
            raise ValueError("record_ids and true_fitness_values must have the same length")

        return [
            self.register_evaluation(
                record_id=record_id,
                true_fitness=fitness,
                evaluation_source=evaluation_source,
            )
            for record_id, fitness in zip(record_ids, true_fitness_values)
        ]

    def pre_records(self, cycle: Optional[int] = None) -> List[PreEvaluationRecord]:
        """Return all pre-evaluation records, optionally for one cycle."""

        records = list(self._pre_records.values())
        if cycle is None:
            return records
        return [record for record in records if record.cycle == cycle]

    def evaluated_records(
        self, cycle: Optional[int] = None
    ) -> List[EvaluatedBeliefRecord]:
        """Return completed records, optionally for one cycle."""

        records = list(self._evaluated_records.values())
        if cycle is None:
            return records
        return [record for record in records if record.cycle == cycle]

    def pending_record_ids(self) -> List[str]:
        """Return selected records that do not yet contain real fitness."""

        return [
            record_id
            for record_id, record in self._pre_records.items()
            if record.selected_for_evaluation and record_id not in self._evaluated_records
        ]

    @staticmethod
    def _make_record_id(
        run_id: str,
        cycle: int,
        individual_id: str,
        architecture_id: str,
    ) -> str:
        """Build a stable identifier for one candidate inside one cycle."""

        return f"{run_id}:{int(cycle)}:{individual_id}:{architecture_id}"

    @staticmethod
    def _append_csv(path: Path, row: Dict[str, object]) -> None:
        """Append one dictionary row and create the CSV header when needed."""

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
    def _finite_float(value: float, name: str) -> float:
        """Validate and return one finite floating-point value."""

        clean = float(value)
        if not math.isfinite(clean):
            raise ValueError(f"{name} must be finite")
        return clean

    @classmethod
    def _non_negative_float(cls, value: float, name: str) -> float:
        """Validate and return one finite non-negative value."""

        clean = cls._finite_float(value, name)
        if clean < 0:
            raise ValueError(f"{name} must be non-negative")
        return clean

    @classmethod
    def _unit_interval(cls, value: float, name: str) -> float:
        """Validate and return one value inside the closed unit interval."""

        clean = cls._finite_float(value, name)
        if not 0.0 <= clean <= 1.0:
            raise ValueError(f"{name} must be inside [0, 1]")
        return clean

    @classmethod
    def _optional_non_negative(
        cls, value: Optional[float], name: str
    ) -> Optional[float]:
        """Validate an optional non-negative value."""

        if value is None:
            return None
        return cls._non_negative_float(value, name)

    @classmethod
    def _optional_unit_interval(
        cls, value: Optional[float], name: str
    ) -> Optional[float]:
        """Validate an optional value inside the closed unit interval."""

        if value is None:
            return None
        return cls._unit_interval(value, name)


def _run_self_test() -> None:
    """Check pre-evaluation logging and later fitness attachment."""

    from .estimator import BeliefEstimate

    encoding = ArchitectureEncoding(
        architecture_id="arch-a",
        architecture_string="ca-densenet-pool",
        individual_id="indi0001",
        length=2,
        module_sequence=["ca-densenet", "pool"],
        base_sequence=["densenet", "pool"],
        attention_sequence=["ca", "none"],
        base_attention_pairs=["ca-densenet", "none-pool"],
        module_counts={"ca-densenet": 1, "pool": 1},
        base_counts={"densenet": 1, "pool": 1},
        attention_counts={"ca": 1, "none": 1},
        module_bigrams=["ca-densenet->pool"],
        pair_bigrams=["ca-densenet->none-pool"],
        numeric_summary={"length": 2.0},
        unit_records=[],
    )
    estimate = BeliefEstimate(
        architecture_id="arch-a",
        belief_mean=0.81,
        evidence_strength=2.4,
        effective_neighbour_count=3.1,
        neighbour_disagreement=0.0004,
        max_similarity=0.86,
        used_neighbour_count=4,
        excluded_exact_match_count=0,
        used_prior_only=False,
        neighbours=[],
    )

    monitor = BeliefCycleMonitor()
    pre = monitor.register_pre_evaluation(
        encoding=encoding,
        estimate=estimate,
        run_id="test",
        cycle=1,
    )
    completed = monitor.register_evaluation(pre.record_id, true_fitness=0.825)

    assert len(monitor.pre_records()) == 1
    assert len(monitor.evaluated_records()) == 1
    assert abs(completed.absolute_error - 0.015) < 1e-12
    assert not monitor.pending_record_ids()

    print("Belief monitoring self-test passed.")
    print(completed.to_dict())


if __name__ == "__main__":
    _run_self_test()
