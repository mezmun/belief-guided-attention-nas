"""
This module reads and validates the belief system configuration.

The configuration is stored in the [belief] section of global.ini. The
current version only prepares the settings. It does not change the genetic
algorithm behaviour.
"""

from __future__ import annotations

import configparser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_ALLOWED_MODES = {"off", "monitor", "guided"}
_ALLOWED_SELECTION_POLICIES = {"quota", "mean_topk", "ucb", "novelty", "random"}
_ALLOWED_CALIBRATION_METHODS = {"none", "ridge", "non_negative_linear"}


@dataclass(frozen=True)
class BeliefConfig:
    """Store validated settings for the belief-guided NAS system."""

    enabled: bool = False
    mode: str = "off"
    warmup_generations: int = 5
    candidate_multiplier: int = 5
    evaluation_budget: int = 20
    kernel_bandwidth: float = 0.25
    selection_policy: str = "quota"
    calibration_method: str = "ridge"
    calibration_update_frequency: int = 1
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
            selection_policy=section.get("selection_policy", fallback="quota").strip().lower(),
            calibration_method=section.get("calibration_method", fallback="ridge").strip().lower(),
            calibration_update_frequency=section.getint(
                "calibration_update_frequency", fallback=1
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

    def validate(self) -> None:
        """Raise a clear error when a configuration value is invalid."""

        if self.mode not in _ALLOWED_MODES:
            raise ValueError(
                f"Invalid belief mode '{self.mode}'. Allowed values: {sorted(_ALLOWED_MODES)}"
            )

        if self.enabled and self.mode == "off":
            raise ValueError("Belief is enabled, but mode is set to 'off'")

        if not self.enabled and self.mode != "off":
            raise ValueError("Belief is disabled, but mode is not set to 'off'")

        if self.warmup_generations < 0:
            raise ValueError("warmup_generations must be zero or greater")

        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")

        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be at least 1")

        if self.kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be greater than zero")

        if self.selection_policy not in _ALLOWED_SELECTION_POLICIES:
            raise ValueError(
                "Invalid selection_policy. "
                f"Allowed values: {sorted(_ALLOWED_SELECTION_POLICIES)}"
            )

        if self.calibration_method not in _ALLOWED_CALIBRATION_METHODS:
            raise ValueError(
                "Invalid calibration_method. "
                f"Allowed values: {sorted(_ALLOWED_CALIBRATION_METHODS)}"
            )

        if self.calibration_update_frequency < 1:
            raise ValueError("calibration_update_frequency must be at least 1")

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
