# Belief-Guided NAS Package

This package contains the new belief-guided evaluation components for the attention-aware evolutionary NAS project.

## Design goal

The existing code will communicate with this package through `BeliefManager`.

## Planned modules

- `config.py`: reads and validates belief settings.
- `manager.py`: provides the main interface for the existing GA code.
- `encoder.py`: converts an architecture into deterministic features.
- `archive.py`: stores unique evaluated architectures and their true fitness.
- `similarity.py`: calculates component-aware architecture similarity.
- `belief.py`: calculates belief mean and uncertainty.
- `selector.py`: selects candidates for real evaluation.
- `metrics.py`: calculates cycle-level validation metrics.
- `storage.py`: saves and restores belief state and logs.

## Current status

The current package is only a scaffold. It reads the `[belief]` section from `global.ini` and validates the settings. It does not change the GA behaviour yet.

## Operating modes

- `off`: the belief system is disabled.
- `monitor`: belief values are recorded, but all normal offspring are still evaluated.
- `guided`: belief values are used to choose which offspring are evaluated.

The project must start with `enabled = 0` and `mode = off` until the integration steps are completed.
