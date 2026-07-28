"""
This module reads and validates the belief system configuration.

All belief settings are stored in the [belief] section of global.ini. The
configuration keeps experimental choices outside the main genetic algorithm
code, so the original search can still be run with the belief system disabled.
"""

from __future__ import annotations

import configparser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_ALLOWED_MODES = {"off", "monitor", "guided"}
_ALLOWED_SELECTION_POLICIES = {"quota", "mean_topk", "ucb", "novelty", "random"}
_ALLOWED_CALIBRATION_METHODS = {"none", "ridge"}
_ALLOWED_BELIEF_METHODS = {"kernel_mean", "bayesian_precision"}


@dataclass(frozen=True)
class BeliefConfig:
    """Store validated settings for the belief-guided NAS system."""

    enabled: bool = False
    mode: str = "off"
    warmup_generations: int = 5
    candidate_multiplier: int = 5
    evaluation_budget: int = 20

    kernel_bandwidth: float = 0.25
    belief_method: str = "kernel_mean"
    top_neighbours: int = 5
    exclude_exact_matches: bool = True
    minimum_archive_size: int = 10

    selection_policy: str = "quota"
    mean_quota: float = 0.60
    ucb_quota: float = 0.20
    novelty_quota: float = 0.10
    random_quota: float = 0.10
    ucb_kappa: float = 0.50
    novelty_neighbours: int = 5

    calibration_method: str = "ridge"
    calibration_min_samples: int = 40
    calibration_update_frequency: int = 1
    calibration_ridge_alpha: float = 1.0
    calibration_random_audit_only: bool = True

    learn_similarity_weights: bool = True
    similarity_min_pairs: int = 50
    similarity_max_pairs: int = 5000
    similarity_target_tau: float = 0.02
    similarity_ridge_alpha: float = 1.0

    random_seed: int = 2312390
    output_directory: str = "belief_outputs"

    @classmethod
    def from_ini(cls, ini_path: Optional[Path] = None) -> "BeliefConfig":
        """Load belief settings from global.ini and validate them."""

        path = ini_path or cls.default_ini_path()
        parser = configparser.ConfigParser()

        if not path.exists():
            raise FileNotFoundError(f"Configuration file was not found: {path}")

        parser.read(path)
        if "belief" not in parser:
            raise KeyError("The [belief] section is missing from global.ini")

        section = parser["belief"]
        config = cls(
            enabled=section.getboolean("enabled", fallback=False),
            mode=section.get("mode", fallback="off").strip().lower(),
            warmup_generations=section.getint("warmup_generations", fallback=5),
            candidate_multiplier=section.getint("candidate_multiplier", fallback=5),
            evaluation_budget=section.getint("evaluation_budget", fallback=20),
            kernel_bandwidth=section.getfloat("kernel_bandwidth", fallback=0.25),
            belief_method=section.get("belief_method", fallback="kernel_mean").strip().lower(),
            top_neighbours=section.getint("top_neighbours", fallback=5),
            exclude_exact_matches=section.getboolean("exclude_exact_matches", fallback=True),
            minimum_archive_size=section.getint("minimum_archive_size", fallback=10),
            selection_policy=section.get("selection_policy", fallback="quota").strip().lower(),
            mean_quota=section.getfloat("mean_quota", fallback=0.60),
            ucb_quota=section.getfloat("ucb_quota", fallback=0.20),
            novelty_quota=section.getfloat("novelty_quota", fallback=0.10),
            random_quota=section.getfloat("random_quota", fallback=0.10),
            ucb_kappa=section.getfloat("ucb_kappa", fallback=0.50),
            novelty_neighbours=section.getint("novelty_neighbours", fallback=5),
            calibration_method=section.get("calibration_method", fallback="ridge").strip().lower(),
            calibration_min_samples=section.getint("calibration_min_samples", fallback=40),
            calibration_update_frequency=section.getint(
                "calibration_update_frequency", fallback=1
            ),
            calibration_ridge_alpha=section.getfloat(
                "calibration_ridge_alpha", fallback=1.0
            ),
            calibration_random_audit_only=section.getboolean(
                "calibration_random_audit_only", fallback=True
            ),
            learn_similarity_weights=section.getboolean(
                "learn_similarity_weights", fallback=True
            ),
            similarity_min_pairs=section.getint("similarity_min_pairs", fallback=50),
            similarity_max_pairs=section.getint("similarity_max_pairs", fallback=5000),
            similarity_target_tau=section.getfloat("similarity_target_tau", fallback=0.02),
            similarity_ridge_alpha=section.getfloat(
                "similarity_ridge_alpha", fallback=1.0
            ),
            random_seed=section.getint("random_seed", fallback=2312390),
            output_directory=section.get("output_directory", fallback="belief_outputs").strip(),
        )
        config.validate()
        return config

    @staticmethod
    def default_ini_path() -> Path:
        """Return the global.ini path from the project root."""

        project_root = Path(__file__).resolve().parents[2]
        return project_root / "global.ini"

    @staticmethod
    def project_root() -> Path:
        """Return the project root used for relative output paths."""

        return Path(__file__).resolve().parents[2]

    def output_path(self) -> Path:
        """Return the absolute belief output directory."""

        path = Path(self.output_directory)
        return path if path.is_absolute() else self.project_root() / path

    def validate(self) -> None:
        """Raise a clear error when a configuration value is invalid."""

        if self.mode not in _ALLOWED_MODES:
            raise ValueError(f"Invalid belief mode '{self.mode}'")
        if self.enabled and self.mode == "off":
            raise ValueError("Belief is enabled, but mode is set to 'off'")
        if not self.enabled and self.mode != "off":
            raise ValueError("Belief is disabled, but mode is not set to 'off'")
        if self.belief_method not in _ALLOWED_BELIEF_METHODS:
            raise ValueError(f"Invalid belief_method '{self.belief_method}'")
        if self.selection_policy not in _ALLOWED_SELECTION_POLICIES:
            raise ValueError(f"Invalid selection_policy '{self.selection_policy}'")
        if self.calibration_method not in _ALLOWED_CALIBRATION_METHODS:
            raise ValueError(f"Invalid calibration_method '{self.calibration_method}'")

        integer_limits = {
            "warmup_generations": (self.warmup_generations, 0),
            "candidate_multiplier": (self.candidate_multiplier, 1),
            "evaluation_budget": (self.evaluation_budget, 1),
            "top_neighbours": (self.top_neighbours, 0),
            "minimum_archive_size": (self.minimum_archive_size, 1),
            "novelty_neighbours": (self.novelty_neighbours, 1),
            "calibration_min_samples": (self.calibration_min_samples, 2),
            "calibration_update_frequency": (self.calibration_update_frequency, 1),
            "similarity_min_pairs": (self.similarity_min_pairs, 1),
            "similarity_max_pairs": (self.similarity_max_pairs, 1),
        }
        for name, (value, minimum) in integer_limits.items():
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")

        positive_values = {
            "kernel_bandwidth": self.kernel_bandwidth,
            "calibration_ridge_alpha": self.calibration_ridge_alpha,
            "similarity_target_tau": self.similarity_target_tau,
            "similarity_ridge_alpha": self.similarity_ridge_alpha,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        if self.ucb_kappa < 0:
            raise ValueError("ucb_kappa must be non-negative")
        if self.similarity_max_pairs < self.similarity_min_pairs:
            raise ValueError("similarity_max_pairs cannot be smaller than similarity_min_pairs")

        quotas = [self.mean_quota, self.ucb_quota, self.novelty_quota, self.random_quota]
        if any(value < 0 for value in quotas) or sum(quotas) <= 0:
            raise ValueError("Selection quotas must be non-negative and have a positive total")
        if not self.output_directory:
            raise ValueError("output_directory cannot be empty")

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""

        return asdict(self)


if __name__ == "__main__":
    loaded_config = BeliefConfig.from_ini()
    print("Belief configuration is valid.")
    for key, value in loaded_config.as_dict().items():
        print(f"{key}: {value}")
