# Test-day confirmation (written 2026-10-24, before any test-split data is read by this repository)

The Stage-0 follow-up (`stage0_followup.md`) returned GO with the repaired design V3: per-copy refinement and
day-before calibration. V3 was selected on the same validation days on which it was judged, so its false-alarm
control and its power advantage are confirmed only by one run on the six unread test days. This file fixes that
run and its decision rule. Nothing here is changed after the run starts; anything computed later that is not
listed here is reported as exploratory.

State of the test split when this was written: no file in this repository has read it. The generator repository
used it once, to confirm the generator (job 32358); this study does not select or tune the generator on it.

## Design (per symbol; six Binance symbols BTC, ETH, SOL, XRP, DOGE, LTC; injection seed 0)

Windows are non-overlapping 100-event stretches with the 100 real events before them as context, as in Stage 0.
Validation days are numbered 1-6 and test days T1-T6.

- **Reference set A (unconditional null):** up to 1,500 windows sampled from validation days 4 and 5, the two
  days before the first calibration day, as in Stage 0. Fixed for the whole run.
- **First calibration day:** all clean windows of validation day 6.
- **Evaluation:** every window of each test day T1-T6, scored clean and in the 13 injected copies of the Stage-0
  grid (spoof k in {5, 10, 20, 50, 100}, tau = 20; stuffing n in {10, 25, 50, 100}; ignition m in {1, 2, 3, 5},
  r = 10), with one position, side and direction per family and window.
- **Calibration (V3, both detectors):** test day T1 is calibrated on validation day 6; day Tk (k >= 2) on the
  clean copies of the windows of day T(k-1).
- **Resolution (V3):** 200 continuations per window; when any copy has a statistic with Monte Carlo
  p <= 2/201, 800 more are drawn, and a copy uses all 1,000 only if that copy itself reached the floor.
- **Everything else as in Stage 0:** the frozen generator `slow-s|W30000`, the seven-statistic panel, the
  Cauchy combination with p capped at 0.99, conformal p-values (1 + #{cal >= score}) / (n_cal + 1).

## Integrity

- The generator files' SHA-256 must equal those stored in the Stage-0 result files
  (`results/pilot_strength_*_seed0.json`, `provenance.file_sha256`). If not, the run is void and is not analysed.
  The run uses the cluster copy of the generator repository, which is the copy Stage 0 and the follow-up ran on.
  (The laptop copy of `shared_resources/lob_data.py` differs from it in the module docstring only.)
- The SHA-256 of this repository's code files used by the run are stored with the results and must match the
  commit that adds this file.
- The run step prints progress only, no detection result. Results are read once all six symbols have finished.
  A symbol that fails for a technical reason is rerun with identical settings; no result of any symbol is looked
  at before all six are complete.

## Measures

Realised false-alarm rate: the share of clean test windows with conformal p <= alpha. Power: the share of
injected copies with p <= alpha, averaged over a family's strength grid; pooled = the mean over the six symbols.
alpha = 0.05 is primary, 0.01 secondary.

## Decision rule

- **C1, validity:** both detectors' realised false-alarm rate at alpha = 0.05 is at most 0.10 on every symbol
  (the tolerance under which V3 was selected).
- **C2, power:** a family is confirmed if the pooled power difference at alpha = 0.05 (conditional minus
  unconditional) is positive and its 95% bootstrap interval excludes 0. The interval resamples test windows
  with replacement within each symbol (2,000 replicates, seed 0; a window and its 13 injected copies move
  together) and treats the calibration sets as fixed. C2 holds if at least 2 of the 3 families are confirmed.

**Outcome.**
- **CONFIRMED:** C1 and C2 hold.
- **NOT CONFIRMED (validity):** C1 fails. Power is reported as found, with the false-alarm rates.
- **NOT CONFIRMED (power):** C1 holds and C2 fails.

The paper reports the outcome, whatever it is, next to the validation results.

## Reported, not part of the decision

- False-alarm rate within nominal: <= 0.05 + 2 SE at each symbol's number of clean test windows; false-alarm
  rate at alpha = 0.01; false-alarm rate by test day.
- Power at alpha = 0.01; per-symbol power; size-matched power at alpha = 0.05 and 0.01 and the paired AUC, as
  in Q1 of the follow-up.
- Benjamini-Hochberg at q = 0.10 on mixtures with 5% prevalence, within each test day and symbol (50 random
  mixtures per variant): power and false-discovery proportion.
- Share of copies scored with 1,000 continuations, clean and injected; run time.

## Not in this run

The proposal's other hypotheses (detection of a held-out pattern against a supervised detector; topological
statistics) and the LOBSTER secondary set are not part of this paper and are not tested here.

## Checks done before the run

- `lobsurv/confirm.py` (day assignment across the validation/test boundary, V3 conformal p-values from raw
  p-values, bootstrap interval) is tested on synthetic inputs (`tests/test_confirm.py`).
- `experiments/confirm_test.py` was smoke-tested in validation mode only (validation days 1-2 reference, day 3
  first calibration, days 4-6 evaluation; 20 windows per day, 50/100 continuations, BN_LTCUSDT). The smoke output
  is not a result and was not kept.
- On BN_LTCUSDT validation data, `part_days` gives the same six days of 998 windows as the Stage-0 day split.
