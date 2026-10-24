# LOB-Surveillance

Generator-calibrated surveillance of limit order books: is this window of order flow ordinary *for this
moment*? A frozen level-1 order-book generator provides the conditional null, conformal calibration on
real clean windows makes the p-values valid whatever the generator gets wrong, and Benjamini–Hochberg
controls the false-discovery rate.

This is the data-science (DS) track. It continues the generator work in the companion repository
[orderflow-noise-models](https://github.com/Arshad-13/orderflow-noise-models) and imports its code instead of copying it.

- Research proposal: [`proposal/proposal.md`](proposal/proposal.md)

## Status (2026-10-21)

Stage 0 has started: the generator bridge, the statistics panel, the FDR tools and the persistence statistic
are written and tested. Two measurements are in the proposal (§6.1): simulation cost, and a queue-size
defect of the generator on LOBSTER data that has to be fixed before LOBSTER can be used. No experiment has
been run yet, and there are no results.

## Layout

```
lobsurv/
  fdr.py           Monte Carlo and conformal p-values, Cauchy combination, BH / BY, FDP and power
  statistics.py    level-1 window statistics (running scores, usable with multilevel splitting)
  null_model.py    GeneratorNull: the companion generator as a conditional null; provenance capture
  persistence.py   sliding-window Rips persistence of log-signature clouds (H0 = MST, H1)
tests/             unit tests; test_null_model.py runs the real generator (opt-in)
proposal/          research proposal
```

## Setup

1. Clone the generator repository next to this one so that the layout is
   `<parent>/Research` (generator) and `<parent>/DS/LOB-Surveillance` (this repo). Otherwise set
   `LOB_GENERATOR_REPO` to its path. The generator repository holds the data loaders and the raw data.
2. Use the generator's environment (`lobresearch-torch`) or install `requirements.txt` into it.
3. Run the tests:

```bash
python -m pytest                          # unit tests
LOBSURV_INTEGRATION=1 python -m pytest    # also fits the generator on BN_BTCUSDT and simulates (~30 s)
```

Every result file must store `lobsurv.null_model.generator_provenance()` (the generator commit and the
number of uncommitted files), so each number can be traced to the generator version that produced it.

## Rules carried over from the generator project

- Never hand-edit a number in a table; regenerate it from the result files.
- The test split is touched once, with the configuration frozen in `config/preregistration.yaml` beforehand.
- The generator configuration (`slow-s`, pool `W30000`) is frozen. Do not tune it here; a generator change
  (such as fix G1) is made and validated in the generator repository and reported as a new version.
- Cite only papers you have seen.
