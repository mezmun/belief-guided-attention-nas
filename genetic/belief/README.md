# Belief-Guided Attention NAS

This package adds an archive-wide fitness belief layer to the existing genetic
NAS code. The package is kept under `genetic/belief/` to reduce changes in the
original project.

## Main idea

Each evaluated architecture provides local fitness evidence for unevaluated
architectures. Architecture similarity controls how strongly each real fitness
value contributes to a candidate.

The default belief mean is:

```text
mu_i = sum(K_ij * fitness_j) / sum(K_ij)
```

The kernel weight is:

```text
K_ij = exp(-(1 - similarity_ij)^2 / (2 * bandwidth^2))
```

An optional `bayesian_precision` mode combines the same evidence through a
Gaussian precision update. It is included as an ablation and should not be
presented as a full physical Kalman filter.

## Package modules

- `config.py`: reads and validates `[belief]` settings.
- `encoder.py`: converts `Individual.units` into deterministic features.
- `archive.py`: stores unique architectures with real fitness.
- `storage.py`: saves archive and calibration state.
- `similarity.py`: calculates interpretable architecture similarity.
- `estimator.py`: calculates belief mean and evidence statistics.
- `calibration.py`: learns uncertainty and similarity weights from past data.
- `uncertainty.py`: calculates raw and calibrated uncertainty.
- `novelty.py`: calculates archive-based exploration novelty.
- `selector.py`: selects candidates for real evaluation.
- `monitoring.py`: stores pre-evaluation and post-evaluation rows.
- `metrics.py`: calculates cycle-wise validation metrics.
- `manager.py`: provides the only integration interface used by `evolve.py`.
- `self_test.py`: runs a small integration test without training a model.

## Search modes

### Off

```ini
enabled = 0
mode = off
```

The original GA behaviour is used.

### Monitor

```ini
enabled = 1
mode = monitor
```

The normal offspring population is evaluated. Belief values are recorded before
training and compared with real fitness after training. Belief does not guide
selection.

### Guided

```ini
enabled = 1
mode = guided
```

Warm-up cycles use the normal GA. After warm-up, a larger candidate pool is
generated and only the selected evaluation batch is trained.

## Uncertainty

Raw uncertainty combines:

- archive fitness variance,
- effective neighbour count,
- local fitness disagreement among similar neighbours.

After enough completed rows, a small ridge model learns expected absolute belief
error. This model predicts uncertainty only; it does not predict architecture
fitness.

Novelty remains separate from uncertainty and is used only for exploration.

## Quota selection

The default guided policy uses:

- 60% high belief mean,
- 20% UCB (`mean + kappa * uncertainty`),
- 10% novelty,
- 10% random audit.

The random audit subset provides a less biased check of belief quality during
guided search.

## Output structure

Each fresh run receives its own directory:

```text
belief_outputs/
  active_run.txt
  run_YYYYMMDD_HHMMSS/
    belief_archive.json
    belief_state.json
    candidate_pre_evaluation.csv
    evaluated_offspring.csv
    cycle_metrics.csv
    selection_summary.csv
```

`active_run.txt` allows an interrupted run to restore the correct belief state.

## Recommended first run

Start with:

```ini
enabled = 1
mode = monitor
belief_method = kernel_mean
```

Check cycle-wise Spearman correlation and uncertainty-error correlation. After
monitoring is stable, change only:

```ini
mode = guided
```

## Self-test

Run from the project root:

```bash
python -m genetic.belief.self_test
```

Expected output:

```text
Belief package integration self-test passed.
```
