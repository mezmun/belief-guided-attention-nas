# Belief-Guided NAS Package

This package contains the new belief-guided evaluation components for the attention-aware evolutionary NAS project.

## Design goal

The new code is kept under `genetic/belief/` to avoid mixing the belief system with the existing genetic algorithm implementation.

The existing code will communicate with this package through `BeliefManager`.

## Current modules

- `config.py`: reads and validates belief settings.
- `manager.py`: provides the main interface for the existing GA code.
- `encoder.py`: converts an `Individual` object into deterministic architecture features.

## Planned modules

- `archive.py`: stores unique evaluated architectures and their true fitness.
- `similarity.py`: calculates component-aware architecture similarity.
- `belief.py`: calculates belief mean and uncertainty.
- `selector.py`: selects candidates for real evaluation.
- `metrics.py`: calculates cycle-level validation metrics.
- `storage.py`: saves and restores belief state and logs.

## Architecture encoder

The encoder reads `Individual.units` without changing the existing population classes. It produces:

- module sequence
- base-module sequence
- attention sequence
- base-attention pairs
- module, base, and attention counts
- ordered transitions
- unit-level numeric parameters
- compact numeric summaries
- the existing architecture UUID and architecture string

The output is deterministic and serializable. Fitness is not used by the encoder.

## Current status

The package can now read the configuration and encode architectures. It still does not change the genetic algorithm behaviour.

## Operating modes

- `off`: the belief system is disabled.
- `monitor`: belief values are recorded, but all normal offspring are still evaluated.
- `guided`: belief values are used to choose which offspring are evaluated.

The project must remain at `enabled = 0` and `mode = off` until the integration steps are completed.

## Encoder self-test

Run this command from the project root:

```bash
python -m genetic.belief.encoder
```

Expected first line:

```text
Architecture encoder self-test passed.
```
