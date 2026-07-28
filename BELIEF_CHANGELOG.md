# Belief System Changelog

## Final integration package

- Added a separate `genetic/belief/` package.
- Added deterministic architecture encoding.
- Added a unique evaluated-architecture archive.
- Added component-aware architecture similarity.
- Added archive-wide kernel belief propagation.
- Added optional Bayesian precision belief update.
- Added evidence strength, effective neighbour count, and disagreement signals.
- Added raw and learned uncertainty estimation.
- Added archive-based novelty estimation.
- Added learned non-negative similarity component weights.
- Added monitor, warm-up, and guided search modes.
- Added quota-based evaluation selection and random audit candidates.
- Added cycle-wise Spearman, Pearson, MAE, RMSE, top-k, and calibration metrics.
- Added persistent archive and calibration state.
- Added oversized unique offspring pool generation after warm-up.
- Integrated the belief manager into `evolve.py`.
- Preserved legacy crossover and mutation behaviour while belief guidance is not active.
- Added a package-level integration self-test.
