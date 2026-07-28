# Belief System Changelog

This file records changes related to the belief-guided NAS extension.

## Package 1 - System scaffold

- Added the `genetic/belief/` package.
- Added validated belief configuration loading.
- Added `BeliefManager` as the future integration interface.
- Added initial documentation.
- The existing GA behaviour is unchanged.

## Package 2 - Architecture encoder

- Added `genetic/belief/encoder.py`.
- Added deterministic feature extraction from `Individual.units`.
- Added module, base, attention, pair, transition, and numeric features.
- Reused the existing architecture UUID and architecture string.
- Added encoder access through `BeliefManager`.
- Added a local encoder self-test.
- The genetic algorithm behaviour is still unchanged.
