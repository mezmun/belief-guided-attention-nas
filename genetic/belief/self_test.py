"""
This module runs a lightweight integration test for the belief package.

The test uses small fake architecture objects. It does not train a neural
network, create population scripts, or change the main genetic algorithm state.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Tuple

from .config import BeliefConfig
from .manager import BeliefManager


class FakeUnit:
    """Provide the minimum unit attributes required by ArchitectureEncoder."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeIndividual:
    """Provide a small Individual-compatible object for package testing."""

    def __init__(self, individual_id: str, architecture_id: str, module_type: int) -> None:
        self.id = individual_id
        self.acc = -1.0
        self._architecture_id = architecture_id
        self.units = [
            FakeUnit(
                type=module_type,
                number=0,
                amount=3,
                in_channel=64,
                out_channel=128,
                reduction_ratio=16,
            ),
            FakeUnit(type=2, number=1, max_or_avg=0.75),
        ]

    def uuid(self) -> Tuple[str, str]:
        return self._architecture_id, f"architecture:{self._architecture_id}"


def run() -> None:
    """Run archive, monitoring, uncertainty, and guided selection checks."""

    with TemporaryDirectory() as directory:
        config = BeliefConfig(
            enabled=True,
            mode="guided",
            warmup_generations=0,
            candidate_multiplier=2,
            evaluation_budget=2,
            minimum_archive_size=2,
            belief_method="bayesian_precision",
            calibration_min_samples=2,
            similarity_min_pairs=1,
            output_directory=str(Path(directory) / "belief_outputs"),
        )
        manager = BeliefManager(config=config)

        initial = [
            FakeIndividual("indi0000", "arch_a", 8),
            FakeIndividual("indi0001", "arch_b", 9),
            FakeIndividual("indi0002", "arch_c", 13),
        ]
        for individual, fitness in zip(initial, [0.80, 0.82, 0.84]):
            individual.acc = fitness
        manager.bootstrap_population(initial, generation=0)

        candidates = [
            FakeIndividual("indi01000", "arch_d", 8),
            FakeIndividual("indi01001", "arch_e", 9),
            FakeIndividual("indi01002", "arch_f", 13),
            FakeIndividual("indi01003", "arch_g", 10),
        ]
        preparation = manager.prepare_cycle(candidates, cycle=1, cache_map={})
        assert len(preparation.selected_individuals) == 2
        first_record = manager.monitor.pre_records(cycle=1)[0]
        
        assert first_record.model_variance is not None
        assert first_record.raw_uncertainty is not None

        for index, individual in enumerate(preparation.selected_individuals):
            individual.acc = 0.815 + index * 0.01
        metrics = manager.post_evaluate(preparation.selected_individuals, cycle=1)

        assert metrics is not None
        assert metrics.evaluated_count == 2
        assert len(manager.archive) == 5
        assert (manager.output_directory / "belief_archive.json").exists()
        assert (manager.output_directory / "candidate_pre_evaluation.csv").exists()

    print("Belief package integration self-test passed.")


if __name__ == "__main__":
    run()
