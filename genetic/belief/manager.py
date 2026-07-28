"""
This module is the single integration point for the belief-guided NAS system.

The main evolution code sends candidate offspring to BeliefManager before real
evaluation and sends evaluated offspring back afterwards. All archive,
similarity, uncertainty, selection, calibration, and logging details remain
inside the belief package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .archive import EvaluatedArchitectureArchive
from .calibration import SimilarityWeightCalibrator, UncertaintyCalibrator
from .config import BeliefConfig
from .encoder import ArchitectureEncoder
from .estimator import SimilarityBeliefEstimator
from .metrics import BeliefMetricsCalculator, CycleMetrics
from .monitoring import BeliefCycleMonitor, EvaluatedBeliefRecord
from .novelty import ArchitectureNovelty
from .selector import BeliefSelector, CandidateAssessment, SelectionDecision
from .similarity import ArchitectureSimilarity, SimilarityWeights
from .storage import ArchiveStorage
from .uncertainty import BeliefUncertaintyEstimator


@dataclass
class CyclePreparation:
    """Store the result of one pre-evaluation candidate processing step."""

    cycle: int
    original_candidate_count: int
    unique_candidate_count: int
    known_candidate_count: int
    assessed_candidate_count: int
    selected_individuals: List[Any]
    selected_record_ids: Dict[str, str]
    selection_reasons: Dict[str, str]


class BeliefManager:
    """Coordinate all belief components through one public interface."""

    VERSION = "1.0-final"

    def __init__(
        self,
        config: Optional[BeliefConfig] = None,
        log: Optional[Any] = None,
        restore_existing: bool = False,
    ) -> None:
        """Create the manager and optionally restore the active run state."""

        self.config = config or BeliefConfig.from_ini()
        self.log = log
        self.base_output_directory = self.config.output_path()
        self.base_output_directory.mkdir(parents=True, exist_ok=True)
        self.active_run_path = self.base_output_directory / "active_run.txt"
        self.run_id = self._resolve_run_id(restore_existing)
        self.output_directory = self.base_output_directory / self.run_id
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.output_directory / "belief_archive.json"
        self.state_path = self.output_directory / "belief_state.json"
        self.pre_csv = self.output_directory / "candidate_pre_evaluation.csv"
        self.evaluated_csv = self.output_directory / "evaluated_offspring.csv"
        self.metrics_csv = self.output_directory / "cycle_metrics.csv"
        self.selection_csv = self.output_directory / "selection_summary.csv"

        self.encoder = ArchitectureEncoder()
        self.archive = EvaluatedArchitectureArchive(self.encoder)
        self.similarity = ArchitectureSimilarity()
        self.estimator = SimilarityBeliefEstimator(
            similarity=self.similarity,
            kernel_bandwidth=self.config.kernel_bandwidth,
            method=self.config.belief_method,
        )
        self.uncertainty_calibrator = UncertaintyCalibrator(
            ridge_alpha=self.config.calibration_ridge_alpha
        )
        self.uncertainty_estimator = BeliefUncertaintyEstimator(
            calibrator=self.uncertainty_calibrator
        )
        self.novelty = ArchitectureNovelty(
            similarity=self.similarity,
            top_k=self.config.novelty_neighbours,
        )
        self.selector = BeliefSelector(self.config.random_seed)
        self.monitor = BeliefCycleMonitor(self.pre_csv, self.evaluated_csv)
        self.metrics = BeliefMetricsCalculator()
        self.similarity_calibrator = SimilarityWeightCalibrator(
            target_tau=self.config.similarity_target_tau,
            ridge_alpha=self.config.similarity_ridge_alpha,
            max_pairs=self.config.similarity_max_pairs,
            random_seed=self.config.random_seed,
        )

        self.last_cycle = -1
        self.calibration_samples: List[Dict[str, float]] = []
        self.current_preparation: Optional[CyclePreparation] = None
        self._loaded_archive = False
        if restore_existing:
            self._restore_state()

    @property
    def is_enabled(self) -> bool:
        """Return True when monitoring or guided selection is enabled."""

        return self.config.enabled

    @property
    def is_monitoring(self) -> bool:
        """Return True when belief is recorded without guided selection."""

        return self.is_enabled and self.config.mode == "monitor"

    @property
    def is_guided(self) -> bool:
        """Return True when guided selection is configured."""

        return self.is_enabled and self.config.mode == "guided"

    def guided_active(self, cycle: int) -> bool:
        """Return True when warm-up is complete and the archive is large enough."""

        return (
            self.is_guided
            and cycle > self.config.warmup_generations
            and len(self.archive) >= self.config.minimum_archive_size
        )

    def candidate_target_size(self, population_size: int, cycle: int) -> int:
        """Return the number of offspring to generate before evaluation selection."""

        if self.guided_active(cycle):
            return int(population_size * self.config.candidate_multiplier)
        return int(population_size)

    def bootstrap_population(
        self,
        individuals: Iterable[Any],
        generation: int,
    ) -> None:
        """Add an already evaluated population to the persistent archive."""

        if not self.is_enabled:
            return
        count_as_measurement = not self._loaded_archive
        for individual in individuals:
            if float(getattr(individual, "acc", -1.0)) < 0:
                continue
            self.archive.add_individual(
                individual=individual,
                generation=generation,
                run_id=self.run_id,
                source="bootstrap",
                count_as_new_measurement=count_as_measurement,
            )
        self._save_state(generation)
        self._info(f"Belief archive bootstrapped with {len(self.archive)} unique architectures")

    def prepare_cycle(
        self,
        candidates: Iterable[Any],
        cycle: int,
        cache_map: Optional[Dict[str, Any]] = None,
    ) -> CyclePreparation:
        """Score candidates, log pre-evaluation data, and choose a real evaluation batch."""

        if not self.is_enabled:
            raise RuntimeError("prepare_cycle was called while the belief system is disabled")
        candidate_list = list(candidates)
        cache = cache_map or {}
        unique: List[tuple[Any, Any]] = []
        seen = set()
        for individual in candidate_list:
            encoding = self.encoder.encode(individual)
            if encoding.architecture_id not in seen:
                seen.add(encoding.architecture_id)
                unique.append((individual, encoding))

        unknown: List[tuple[Any, Any]] = []
        known_count = 0
        for individual, encoding in unique:
            if self.archive.contains(encoding.architecture_id):
                individual.acc = self.archive.get(encoding.architecture_id).fitness_mean
                known_count += 1
                continue
            if encoding.architecture_id in cache:
                individual.acc = float(cache[encoding.architecture_id])
                self.archive.add_individual(
                    individual=individual,
                    generation=cycle,
                    run_id=self.run_id,
                    source="cache",
                    count_as_new_measurement=False,
                )
                known_count += 1
                continue
            unknown.append((individual, encoding))

        assessments: List[CandidateAssessment] = []
        if len(self.archive) > 0:
            for individual, encoding in unknown:
                belief = self.estimator.estimate_one(
                    candidate=encoding,
                    archive=self.archive,
                    top_neighbours=self.config.top_neighbours,
                    exclude_exact_match=self.config.exclude_exact_matches,
                )
                uncertainty = self.uncertainty_estimator.estimate(belief, self.archive)
                novelty = self.novelty.estimate(encoding, self.archive)
                assessments.append(
                    CandidateAssessment(
                        individual=individual,
                        encoding=encoding,
                        belief=belief,
                        uncertainty=uncertainty,
                        novelty=novelty,
                    )
                )

        decisions = self._select_assessments(assessments, cycle)
        decision_by_architecture = {
            item.assessment.encoding.architecture_id: item for item in decisions
        }
        selected_individuals = (
            [item.assessment.individual for item in decisions]
            if self.guided_active(cycle)
            else candidate_list
        )
        selected_record_ids: Dict[str, str] = {}
        selection_reasons: Dict[str, str] = {}

        for assessment in assessments:
            architecture_id = assessment.encoding.architecture_id
            decision = decision_by_architecture.get(architecture_id)
            selected = decision is not None
            reason = decision.reason if decision else "not_selected"
            score = decision.selection_score if decision else None
            record = self.monitor.register_pre_evaluation(
                encoding=assessment.encoding,
                estimate=assessment.belief,
                run_id=self.run_id,
                cycle=cycle,
                selected_for_evaluation=selected,
                selection_reason=reason,
                belief_uncertainty=assessment.uncertainty.uncertainty,
                raw_uncertainty=assessment.uncertainty.raw_uncertainty,
                novelty=assessment.novelty.novelty,
                selection_score=score,
            )
            if selected:
                selected_record_ids[assessment.encoding.individual_id] = record.record_id
                selection_reasons[assessment.encoding.individual_id] = reason

        preparation = CyclePreparation(
            cycle=cycle,
            original_candidate_count=len(candidate_list),
            unique_candidate_count=len(unique),
            known_candidate_count=known_count,
            assessed_candidate_count=len(assessments),
            selected_individuals=selected_individuals,
            selected_record_ids=selected_record_ids,
            selection_reasons=selection_reasons,
        )
        self.current_preparation = preparation
        self._write_selection_summary(preparation)
        self._info(
            "Belief cycle %d: candidates=%d, unique=%d, known=%d, selected=%d"
            % (
                cycle,
                len(candidate_list),
                len(unique),
                known_count,
                len(selected_individuals),
            )
        )
        return preparation

    def post_evaluate(
        self,
        evaluated_individuals: Iterable[Any],
        cycle: int,
    ) -> Optional[CycleMetrics]:
        """Attach true fitness, update the archive, calibrate, and save state."""

        if not self.is_enabled:
            return None
        if self.current_preparation is None or self.current_preparation.cycle != cycle:
            raise RuntimeError("No matching belief preparation exists for this cycle")

        completed: List[EvaluatedBeliefRecord] = []
        for individual in evaluated_individuals:
            individual_id = str(getattr(individual, "id", "unknown"))
            record_id = self.current_preparation.selected_record_ids.get(individual_id)
            fitness = float(getattr(individual, "acc", -1.0))
            if fitness < 0:
                raise ValueError(f"Selected individual has no real fitness: {individual_id}")

            if record_id is not None:
                record = self.monitor.register_evaluation(
                    record_id=record_id,
                    true_fitness=fitness,
                    evaluation_source="training",
                )
                completed.append(record)
                if self._use_for_uncertainty_calibration(record):
                    self.calibration_samples.append(
                        {
                            "evidence_strength": record.evidence_strength,
                            "neighbour_disagreement": record.neighbour_disagreement,
                            "effective_neighbour_count": record.effective_neighbour_count,
                            "absolute_error": record.absolute_error,
                        }
                    )

            self.archive.add_individual(
                individual=individual,
                generation=cycle,
                run_id=self.run_id,
                source="training" if record_id is not None else "inherited_or_duplicate",
                count_as_new_measurement=record_id is not None,
            )

        cycle_metrics = None
        if completed:
            cycle_metrics = self.metrics.calculate_cycle(
                completed, top_k=min(5, len(completed))
            )
            BeliefCycleMonitor.append_csv(self.metrics_csv, cycle_metrics.to_dict())

        self._update_calibration(cycle)
        self.last_cycle = cycle
        self._save_state(cycle)
        self.current_preparation = None
        return cycle_metrics

    def describe(self) -> Dict[str, Any]:
        """Return a compact status summary for logs."""

        return {
            "enabled": self.is_enabled,
            "mode": self.config.mode,
            "run_id": self.run_id,
            "archive_size": len(self.archive),
            "belief_method": self.config.belief_method,
            "similarity_weights": asdict(self.similarity.weights),
            "uncertainty_calibrated": self.uncertainty_calibrator.state.fitted,
        }

    def _select_assessments(
        self,
        assessments: List[CandidateAssessment],
        cycle: int,
    ) -> List[SelectionDecision]:
        """Choose all warm-up candidates or a guided evaluation subset."""

        if not assessments:
            return []
        if not self.guided_active(cycle):
            reason = "monitor_all" if self.is_monitoring else "warmup_all"
            return [
                SelectionDecision(
                    assessment=item,
                    reason=reason,
                    selection_score=item.belief.belief_mean,
                )
                for item in assessments
            ]
        return self.selector.select(
            assessments=assessments,
            budget=self.config.evaluation_budget,
            policy=self.config.selection_policy,
            ucb_kappa=self.config.ucb_kappa,
            quotas=(
                self.config.mean_quota,
                self.config.ucb_quota,
                self.config.novelty_quota,
                self.config.random_quota,
            ),
        )

    def _use_for_uncertainty_calibration(self, record: EvaluatedBeliefRecord) -> bool:
        """Use full warm-up rows and optional random audit rows after warm-up."""

        if record.cycle <= self.config.warmup_generations:
            return True
        if not self.config.calibration_random_audit_only:
            return True
        return record.selection_reason == "random_audit"

    def _update_calibration(self, cycle: int) -> None:
        """Update learned uncertainty and similarity values from past data."""

        if cycle % self.config.calibration_update_frequency != 0:
            return
        if (
            self.config.calibration_method == "ridge"
            and len(self.calibration_samples) >= self.config.calibration_min_samples
        ):
            self.uncertainty_calibrator.fit_samples(self.calibration_samples)

        pair_count = len(self.archive) * (len(self.archive) - 1) // 2
        if (
            self.config.learn_similarity_weights
            and pair_count >= self.config.similarity_min_pairs
        ):
            learned = self.similarity_calibrator.fit(self.archive, self.similarity)
            if learned is not None:
                self.similarity.weights = learned.normalized()

    def _restore_state(self) -> None:
        """Restore archive and calibration state when files are available."""

        if self.archive_path.exists():
            self.archive = ArchiveStorage.load_archive(self.archive_path, self.encoder)
            self._loaded_archive = True
        if not self.state_path.exists():
            return
        state = ArchiveStorage.load_state(self.state_path)
        stored_run_id = str(state.get("run_id", self.run_id))
        if stored_run_id != self.run_id:
            raise ValueError("Belief state run_id does not match the active run")
        self.last_cycle = int(state.get("last_cycle", -1))
        self.calibration_samples = [
            {key: float(value) for key, value in item.items()}
            for item in state.get("calibration_samples", [])
        ]
        uncertainty_state = state.get("uncertainty_calibration")
        if isinstance(uncertainty_state, dict):
            self.uncertainty_calibrator.load_state(uncertainty_state)
        weight_data = state.get("similarity_weights")
        if isinstance(weight_data, dict):
            self.similarity.weights = SimilarityWeights(
                **{key: float(value) for key, value in weight_data.items()}
            ).normalized()

    def _save_state(self, cycle: int) -> None:
        """Save archive, learned values, and compact calibration samples."""

        ArchiveStorage.save_archive(self.archive, self.archive_path)
        ArchiveStorage.save_state(
            {
                "run_id": self.run_id,
                "last_cycle": int(cycle),
                "uncertainty_calibration": self.uncertainty_calibrator.state.to_dict(),
                "calibration_samples": self.calibration_samples[-10000:],
                "similarity_weights": asdict(self.similarity.weights),
                "similarity_pair_count": self.similarity_calibrator.last_pair_count,
                "config": self.config.as_dict(),
            },
            self.state_path,
        )

    def _write_selection_summary(self, preparation: CyclePreparation) -> None:
        """Append one compact summary of the candidate selection step."""

        BeliefCycleMonitor.append_csv(
            self.selection_csv,
            {
                "run_id": self.run_id,
                "cycle": preparation.cycle,
                "guided_active": self.guided_active(preparation.cycle),
                "original_candidate_count": preparation.original_candidate_count,
                "unique_candidate_count": preparation.unique_candidate_count,
                "known_candidate_count": preparation.known_candidate_count,
                "assessed_candidate_count": preparation.assessed_candidate_count,
                "selected_count": len(preparation.selected_individuals),
            },
        )

    def _info(self, message: str) -> None:
        """Write a message through the existing logger when available."""

        if self.log is not None and hasattr(self.log, "info"):
            self.log.info(message)

    def _resolve_run_id(self, restore_existing: bool) -> str:
        """Create a new run directory or restore the currently active one."""

        if restore_existing and self.active_run_path.exists():
            run_id = self.active_run_path.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id
        run_id = self._new_run_id()
        self.active_run_path.write_text(run_id, encoding="utf-8")
        return run_id

    @staticmethod
    def _new_run_id() -> str:
        """Create a readable UTC run identifier."""

        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
