# LOB-Surveillance

Generator-calibrated surveillance of limit order books: is this window of order flow ordinary *for this
moment*? A frozen level-1 order-book generator simulates continuations of each 100-event window from its own
real context, Monte Carlo p-values of seven statistics are combined into one score, and split-conformal
calibration on real clean windows turns the score into a p-value that is valid under exchangeability whatever
the generator gets wrong. The comparison is with a detector that differs only in its null: one pooled,
unconditional reference set.

Paper: [`manuscript/main.pdf`](manuscript/main.pdf) (IEEE double column; source `manuscript/main.tex`).

This is the data-science (DS) track. It continues the generator work in the companion repository
[orderflow-noise-models](https://github.com/Arshad-13/orderflow-noise-models) and imports its code instead of
copying it.

## Results

Six Binance USD-M perpetual futures (BTC, ETH, SOL, XRP, DOGE, LTC; February 2024), with spoofing, quote
stuffing and momentum ignition injected at 13 strengths. Every decision rule was committed before the analysis
it governs; each result is appended to its rule file.

| Step | Rule (commit) | Data | Outcome |
| --- | --- | --- | --- |
| Stage 0 | [`stage0_redesign.md`](results/stage0_redesign.md) (818a11d) | validation | **NO-GO**: conditional power higher in all three families, but its false-alarm rate at α = 0.05 reached 0.110 on XRP (limit 0.10) |
| Follow-up | [`stage0_followup.md`](results/stage0_followup.md) (ef41139) | validation | **GO** with repaired design V3: refinement matched between calibration and evaluation windows, each day calibrated on the day before; largest false-alarm rate 0.075, power still higher in all three families |
| Test days | [`test_confirmation.md`](results/test_confirmation.md) (387dbd4) | test, read once | **NOT CONFIRMED (validity)**: the *unconditional* detector exceeded the limit on SOL (0.140) and DOGE (0.232); the conditional detector stayed at 0.039–0.052 on every symbol. Power higher for spoofing and stuffing, lower for momentum ignition |

What the follow-up found: the excess false alarms came partly from adaptive Monte Carlo refinement that a
window's injected copies could trigger (so clean evaluation windows were scored at higher resolution than
calibration windows) and mostly from drift between the calibration day and the evaluation days.

Not decided by any rule, and reported in the paper as exploratory: momentum ignition reversed between validation
and test days, and the test-day bootstrap holds the six symbols fixed, so its intervals are too narrow.
[`stage0_pilot_plan.md`](results/stage0_pilot_plan.md) is the earlier pilot whose primary measure
(Benjamini–Hochberg power) was replaced before Stage 0; the paper says why.

## Reproduce

Cluster-sized runs on 16 CPU cores: about 2 hours per symbol for Stage 0 and 6–8 hours for the test-day run.
Every result file stores the generator's file hashes (`lobsurv.null_model.generator_provenance()`).

```bash
# Stage 0 (validation), per symbol; --save-raw is the follow-up's rerun with per-statistic p-values
python -m experiments.pilot_strength --dataset BN_BTCUSDT [--save-raw]
# Follow-up: Q1 from the saved files, Q2 from the rerun
python -m experiments.followup q1
python -m experiments.followup q2
# Test-day confirmation, per symbol, then the decision over all symbols
python -m experiments.confirm_test run --dataset BN_BTCUSDT --split test
python -m experiments.confirm_test decide --split test
# Every number and table of the paper
python -m experiments.paper_assets
```

Build the paper with any LaTeX engine from `manuscript/` (`main.tex` reads `generated/`).

## Layout

```
lobsurv/
  statistics.py    level-1 window statistics
  null_model.py    GeneratorNull: the companion generator as a conditional null; provenance capture
  scoring.py       window scores of both detectors (Monte Carlo p-values, Cauchy combination, refinement)
  injection.py     spoofing, quote stuffing and momentum ignition footprints written into real windows
  fdr.py           Monte Carlo and conformal p-values, Benjamini-Hochberg, FDP and power
  followup.py      size-matched power, AUC and the V1-V3 arms of the follow-up
  confirm.py       test-day day assignment, rolling calibration and the bootstrap of the decision rule
  persistence.py   persistence statistic of the proposal (not used in the paper)
experiments/       Stage 0, follow-up, test-day confirmation, paper_assets
results/           rule files (*.md), result files (*.json), raw per-statistic p-values (raw/*.npz)
manuscript/        main.tex, main.pdf, generated/ (number macros and tables)
proposal/          research proposal
tests/             unit tests; test_null_model.py runs the real generator (opt-in)
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

## Rules carried over from the generator project

- Never hand-edit a number in a table; regenerate it from the result files.
- The test split is read once, by a run whose rule and code were committed beforehand.
- The generator configuration (`slow-s`, pool `W30000`) is frozen. Do not tune it here; a generator change
  is made and validated in the generator repository and reported as a new version.
- Cite only papers you have seen.
