"""
This module stores architectures with known real fitness values.

The archive keeps one record for each unique architecture. Repeated evaluations
of the same architecture are combined inside the same record, so exact
duplicates do not act as independent neighbours during belief calculation.
"""

from __future__ import annotations

import math
from numbers import Real
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .encoder import ArchitectureEncoder, ArchitectureEncoding


@dataclass
class ArchiveEntry:
    """Store one unique evaluated architecture and its fitness history."""

    architecture_id: str
    architecture_string: str
    encoding: ArchitectureEncoding
    fitness_mean: float
    fitness_last: float
    fitness_best: float
    fitness_worst: float
    evaluation_count: int
    first_generation: int
    last_generation: int
    first_individual_id: str
    last_individual_id: str
    run_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)

    def add_measurement(
        self,
        fitness: float,
        generation: int,
        individual_id: str,
        run_id: str,
        source: str,
    ) -> None:
        """Add one new real fitness measurement to this unique architecture."""

        self.fitness_history.append(float(fitness))
        self.evaluation_count += 1
        self.fitness_last = float(fitness)
        self.fitness_best = max(self.fitness_best, float(fitness))
        self.fitness_worst = min(self.fitness_worst, float(fitness))
        self.fitness_mean = sum(self.fitness_history) / len(self.fitness_history)
        self.last_generation = int(generation)
        self.last_individual_id = str(individual_id)
        self._append_unique(self.run_ids, run_id)
        self._append_unique(self.sources, source)

    def mark_seen(
        self,
        generation: int,
        individual_id: str,
        run_id: str,
        source: str,
    ) -> None:
        """Record a repeated sighting without adding a new fitness measurement."""

        self.last_generation = int(generation)
        self.last_individual_id = str(individual_id)
        self._append_unique(self.run_ids, run_id)
        self._append_unique(self.sources, source)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary for later storage."""

        data = asdict(self)
        data["encoding"] = self.encoding.to_dict()
        return data

    @staticmethod
    def _append_unique(values: List[str], value: str) -> None:
        """Append a non-empty value only when it is not already present."""

        clean_value = str(value).strip()
        if clean_value and clean_value not in values:
            values.append(clean_value)


@dataclass(frozen=True)
class ArchiveUpdateResult:
    """Describe what happened when an architecture was sent to the archive."""

    architecture_id: str
    status: str
    archive_size: int
    evaluation_count: int
    fitness_mean: float


class EvaluatedArchitectureArchive:
    """Manage unique architectures with known real fitness values."""

    def __init__(self, encoder: Optional[ArchitectureEncoder] = None) -> None:
        """Create an empty archive with a deterministic architecture encoder."""

        self.encoder = encoder or ArchitectureEncoder()
        self._entries: Dict[str, ArchiveEntry] = {}

    def add_individual(
        self,
        individual: Any,
        generation: int,
        run_id: str = "",
        source: str = "training",
        count_as_new_measurement: bool = True,
    ) -> ArchiveUpdateResult:
        """Encode an evaluated Individual and add it to the archive."""

        fitness = self._read_fitness(individual)
        encoding = self.encoder.encode(individual)
        individual_id = str(getattr(individual, "id", "unknown"))

        return self.add_encoding(
            encoding=encoding,
            fitness=fitness,
            generation=generation,
            individual_id=individual_id,
            run_id=run_id,
            source=source,
            count_as_new_measurement=count_as_new_measurement,
        )

    def add_many(
        self,
        individuals: Iterable[Any],
        generation: int,
        run_id: str = "",
        source: str = "training",
        count_as_new_measurement: bool = True,
    ) -> List[ArchiveUpdateResult]:
        """Add several evaluated individuals in their current order."""

        return [
            self.add_individual(
                individual=individual,
                generation=generation,
                run_id=run_id,
                source=source,
                count_as_new_measurement=count_as_new_measurement,
            )
            for individual in individuals
        ]

    def add_encoding(
        self,
        encoding: ArchitectureEncoding,
        fitness: float,
        generation: int,
        individual_id: str = "unknown",
        run_id: str = "",
        source: str = "training",
        count_as_new_measurement: bool = True,
    ) -> ArchiveUpdateResult:
        """Add an already encoded architecture and its real fitness value."""

        clean_fitness = self._validate_fitness(fitness)
        architecture_id = encoding.architecture_id

        if architecture_id not in self._entries:
            entry = ArchiveEntry(
                architecture_id=architecture_id,
                architecture_string=encoding.architecture_string,
                encoding=deepcopy(encoding),
                fitness_mean=clean_fitness,
                fitness_last=clean_fitness,
                fitness_best=clean_fitness,
                fitness_worst=clean_fitness,
                evaluation_count=1,
                first_generation=int(generation),
                last_generation=int(generation),
                first_individual_id=str(individual_id),
                last_individual_id=str(individual_id),
                run_ids=[],
                sources=[],
                fitness_history=[clean_fitness],
            )
            entry._append_unique(entry.run_ids, run_id)
            entry._append_unique(entry.sources, source)
            self._entries[architecture_id] = entry
            status = "added"
        else:
            entry = self._entries[architecture_id]
            self._check_architecture_string(entry, encoding)

            if count_as_new_measurement:
                entry.add_measurement(
                    fitness=clean_fitness,
                    generation=generation,
                    individual_id=individual_id,
                    run_id=run_id,
                    source=source,
                )
                status = "measurement_updated"
            else:
                entry.mark_seen(
                    generation=generation,
                    individual_id=individual_id,
                    run_id=run_id,
                    source=source,
                )
                status = "seen_without_measurement"

        return ArchiveUpdateResult(
            architecture_id=architecture_id,
            status=status,
            archive_size=len(self),
            evaluation_count=entry.evaluation_count,
            fitness_mean=entry.fitness_mean,
        )

    def contains(self, architecture_id: str) -> bool:
        """Return True when the architecture is already stored."""

        return str(architecture_id) in self._entries

    def get(self, architecture_id: str) -> ArchiveEntry:
        """Return one archive entry or raise a clear KeyError."""

        key = str(architecture_id)
        if key not in self._entries:
            raise KeyError(f"Architecture was not found in the archive: {key}")
        return self._entries[key]

    def entries(self) -> List[ArchiveEntry]:
        """Return archive entries in insertion order."""

        return list(self._entries.values())

    def encodings(self) -> List[ArchitectureEncoding]:
        """Return one encoding for each unique architecture."""

        return [entry.encoding for entry in self._entries.values()]

    def fitness_values(self) -> List[float]:
        """Return the mean real fitness of each unique architecture."""

        return [entry.fitness_mean for entry in self._entries.values()]

    def architecture_ids(self) -> List[str]:
        """Return all unique architecture identifiers."""

        return list(self._entries.keys())

    def to_records(self) -> List[Dict[str, Any]]:
        """Return all entries as serializable dictionaries."""

        return [entry.to_dict() for entry in self._entries.values()]

    def summary(self) -> Dict[str, Any]:
        """Return a compact archive summary for logs and tests."""

        total_measurements = sum(entry.evaluation_count for entry in self._entries.values())
        repeated_measurements = total_measurements - len(self._entries)

        return {
            "unique_architectures": len(self._entries),
            "total_measurements": total_measurements,
            "repeated_measurements": repeated_measurements,
            "mean_fitness": self._safe_mean(self.fitness_values()),
            "best_fitness": max(self.fitness_values()) if self._entries else None,
            "worst_fitness": min(self.fitness_values()) if self._entries else None,
        }

    def clear(self) -> None:
        """Remove all entries from the archive."""

        self._entries.clear()

    def __len__(self) -> int:
        """Return the number of unique architectures."""

        return len(self._entries)

    @staticmethod
    def _read_fitness(individual: Any) -> float:
        """Read and validate the real fitness stored in Individual.acc."""

        if not hasattr(individual, "acc"):
            raise AttributeError("The individual does not contain an 'acc' attribute")
        return EvaluatedArchitectureArchive._validate_fitness(getattr(individual, "acc"))

    @staticmethod
    def _validate_fitness(fitness: Any) -> float:
        """Accept a finite non-negative real fitness value."""

        if isinstance(fitness, bool) or not isinstance(fitness, Real):
            raise TypeError("Fitness must be a numeric value")

        clean_fitness = float(fitness)
        if not math.isfinite(clean_fitness):
            raise ValueError("Fitness must be finite")
        if clean_fitness < 0:
            raise ValueError("Fitness must be non-negative and come from a real evaluation")
        return clean_fitness

    @staticmethod
    def _check_architecture_string(
        existing_entry: ArchiveEntry,
        new_encoding: ArchitectureEncoding,
    ) -> None:
        """Detect an unexpected hash collision or inconsistent encoding."""

        if existing_entry.architecture_string != new_encoding.architecture_string:
            raise ValueError(
                "The same architecture_id was produced for two different architecture strings"
            )

    @staticmethod
    def _safe_mean(values: List[float]) -> Optional[float]:
        """Return the mean of a list, or None when the list is empty."""

        if not values:
            return None
        return float(sum(values) / len(values))


def _run_self_test() -> None:
    """Run a small local test without changing the genetic algorithm."""

    class FakeUnit:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FakeIndividual:
        def __init__(self, architecture_id: str, fitness: float, individual_id: str) -> None:
            self.id = individual_id
            self.acc = fitness
            self._architecture_id = architecture_id
            self.units = [
                FakeUnit(type=13, number=0, amount=4, k=20, max_input_channel=128,
                         in_channel=64, out_channel=144, reduction_ratio=16),
                FakeUnit(type=2, number=1, max_or_avg=0.75),
                FakeUnit(type=9, number=2, amount=3, in_channel=144,
                         out_channel=256, reduction_ratio=16),
            ]

        def uuid(self) -> tuple[str, str]:
            return self._architecture_id, "[ca-densenet]-[pool]-[cbam-resnet]"

    archive = EvaluatedArchitectureArchive()
    first = FakeIndividual("same_hash", 0.81, "indi0001")
    repeated = FakeIndividual("same_hash", 0.83, "indi0101")

    result_1 = archive.add_individual(first, generation=0, run_id="test_run")
    result_2 = archive.add_individual(repeated, generation=1, run_id="test_run")

    assert result_1.status == "added"
    assert result_2.status == "measurement_updated"
    assert len(archive) == 1
    assert archive.get("same_hash").evaluation_count == 2
    assert abs(archive.get("same_hash").fitness_mean - 0.82) < 1e-12

    print("Evaluated architecture archive self-test passed.")
    print(archive.summary())


if __name__ == "__main__":
    _run_self_test()
