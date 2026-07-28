"""
This module selects which candidate architectures receive real evaluation.

The selector supports pure ranking policies and a quota policy that balances
high expected fitness, uncertainty-aware exploration, architecture novelty,
and a small random audit subset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .encoder import ArchitectureEncoding
from .estimator import BeliefEstimate
from .novelty import NoveltyEstimate
from .uncertainty import UncertaintyEstimate


@dataclass(frozen=True)
class CandidateAssessment:
    """Combine one candidate object with all pre-evaluation belief signals."""

    individual: Any
    encoding: ArchitectureEncoding
    belief: BeliefEstimate
    uncertainty: UncertaintyEstimate
    novelty: NoveltyEstimate

    @property
    def ucb_score(self) -> float:
        """Return the default UCB score without a policy multiplier."""

        return self.belief.belief_mean + self.uncertainty.uncertainty


@dataclass(frozen=True)
class SelectionDecision:
    """Store one selected candidate and the reason for its selection."""

    assessment: CandidateAssessment
    reason: str
    selection_score: float


class BeliefSelector:
    """Select a unique evaluation batch from candidate assessments."""

    def __init__(self, random_seed: int = 2312390) -> None:
        """Create a deterministic random generator for audit selection."""

        self.random = random.Random(int(random_seed))

    def select(
        self,
        assessments: Iterable[CandidateAssessment],
        budget: int,
        policy: str,
        ucb_kappa: float,
        quotas: Sequence[float],
    ) -> List[SelectionDecision]:
        """Select candidates according to one configured policy."""

        items = self._unique(list(assessments))
        if budget < 1:
            raise ValueError("budget must be at least 1")
        if not items:
            return []
        budget = min(int(budget), len(items))

        if policy == "mean_topk":
            return self._ranked(items, budget, "mean_topk", lambda item: item.belief.belief_mean)
        if policy == "ucb":
            return self._ranked(
                items,
                budget,
                "ucb",
                lambda item: item.belief.belief_mean + ucb_kappa * item.uncertainty.uncertainty,
            )
        if policy == "novelty":
            return self._ranked(items, budget, "novelty", lambda item: item.novelty.novelty)
        if policy == "random":
            chosen = self.random.sample(items, budget)
            return [SelectionDecision(item, "random_audit", 0.0) for item in chosen]
        if policy != "quota":
            raise ValueError(f"Unsupported selection policy: {policy}")

        return self._quota_select(items, budget, ucb_kappa, quotas)

    def _quota_select(
        self,
        items: List[CandidateAssessment],
        budget: int,
        ucb_kappa: float,
        quotas: Sequence[float],
    ) -> List[SelectionDecision]:
        """Select disjoint subsets using normalized largest-remainder quotas."""

        if len(quotas) != 4:
            raise ValueError("quota selection requires four quota values")
        counts = self._quota_counts(budget, quotas)
        selected: Dict[str, SelectionDecision] = {}

        ranked_groups: List[Tuple[str, List[CandidateAssessment], Any]] = [
            (
                "mean_topk",
                sorted(items, key=lambda item: item.belief.belief_mean, reverse=True),
                lambda item: item.belief.belief_mean,
            ),
            (
                "ucb",
                sorted(
                    items,
                    key=lambda item: item.belief.belief_mean
                    + ucb_kappa * item.uncertainty.uncertainty,
                    reverse=True,
                ),
                lambda item: item.belief.belief_mean
                + ucb_kappa * item.uncertainty.uncertainty,
            ),
            (
                "novelty",
                sorted(items, key=lambda item: item.novelty.novelty, reverse=True),
                lambda item: item.novelty.novelty,
            ),
        ]

        for group_index, (reason, ordered, score_fn) in enumerate(ranked_groups):
            needed = counts[group_index]
            for item in ordered:
                key = item.encoding.architecture_id
                if key in selected:
                    continue
                selected[key] = SelectionDecision(item, reason, float(score_fn(item)))
                if sum(decision.reason == reason for decision in selected.values()) >= needed:
                    break

        remaining = [item for item in items if item.encoding.architecture_id not in selected]
        random_needed = min(counts[3], len(remaining))
        for item in self.random.sample(remaining, random_needed):
            selected[item.encoding.architecture_id] = SelectionDecision(
                item, "random_audit", 0.0
            )

        if len(selected) < budget:
            remaining = [item for item in items if item.encoding.architecture_id not in selected]
            ordered = sorted(remaining, key=lambda item: item.belief.belief_mean, reverse=True)
            for item in ordered[: budget - len(selected)]:
                selected[item.encoding.architecture_id] = SelectionDecision(
                    item, "mean_fill", item.belief.belief_mean
                )

        return list(selected.values())[:budget]

    @staticmethod
    def _ranked(
        items: List[CandidateAssessment],
        budget: int,
        reason: str,
        score_fn: Any,
    ) -> List[SelectionDecision]:
        """Return the top candidates under one score function."""

        ordered = sorted(items, key=score_fn, reverse=True)[:budget]
        return [SelectionDecision(item, reason, float(score_fn(item))) for item in ordered]

    @staticmethod
    def _unique(items: List[CandidateAssessment]) -> List[CandidateAssessment]:
        """Keep the first candidate for each architecture identifier."""

        result: List[CandidateAssessment] = []
        seen = set()
        for item in items:
            key = item.encoding.architecture_id
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _quota_counts(budget: int, quotas: Sequence[float]) -> List[int]:
        """Convert arbitrary non-negative quotas into counts that sum to budget."""

        total = float(sum(quotas))
        if total <= 0 or any(value < 0 for value in quotas):
            raise ValueError("quotas must be non-negative and have a positive total")
        raw = [budget * value / total for value in quotas]
        counts = [int(value) for value in raw]
        remaining = budget - sum(counts)
        order = sorted(range(len(raw)), key=lambda index: raw[index] - counts[index], reverse=True)
        for index in order[:remaining]:
            counts[index] += 1
        return counts
