# Stage-0 power pilot: plan and go/no-go rule (written 2026-10-23, before the full run)

Validation split only (the test split is not read). Script: experiments/pilot_power.py.
Six Binance symbols, seed 0; per symbol 600 reference windows (validation days 1-2), 600 calibration windows
(day 3), 600 evaluation windows (days 4-6); 200 generator continuations per window; injections at mid
strength (spoof k = 10, tau = 20; stuffing n = 25; ignition m = 3, r = 10); BH at q = 0.10 on mixtures with
5% prevalence (50 random mixtures).

Development so far, on smoke tests of BN_BTCUSDT validation windows (60/60/60, 50 continuations):
- Cauchy combination capped at p = 0.99 (p = 1 from discrete statistics swamped the combination);
- randomised tie-breaking for both detectors' per-statistic p-values (ties made them strongly conservative);
- n_fast counts events at or below 0.5 ms on millisecond-stamped data (the 5% quantile was the floor value).

Go/no-go (proposal Section 7, Stage 0): GO if the generator-conditional detector has higher mean power than the
unconditional detector (averaged over the six symbols) on at least 2 of the 3 families, with mean empirical
FDR <= 0.15 for both. Otherwise the study is reported as a documented negative result (or redesigned on
validation, with that redesign stated).

## Result (2026-10-23, cluster job 32703; generator code identical to orderflow-noise-models 4513b46 by file hash)

By the rule above: NO-GO. But the rule was uninformative: BH made no discovery for either detector on any
symbol or family (power 0.00, FDR 0.00 everywhere). With 600 calibration windows the smallest conformal p is
1/601, and with 600 test windows BH at q = 0.10 needs at least 10 windows scoring above every calibration window
before it can reject anything; at mid strength and 5% prevalence that never happened. This is a design error of
the pilot (the proposal's feasibility section already asked for about 3,000 calibration windows), not evidence
about the detectors.

Informative secondary measure, share of injected windows with conformal p <= 0.01 (clean windows: 0.5-2.2%):
- ignition: generator 1-48% (LTC 48, XRP 28, DOGE 17, BTC 6, ETH 5, SOL 1); unconditional 1-2% (XRP 26).
- spoof: generator 1-4%; unconditional 1-3% (SOL 14).
- stuffing: generator 1-4%; unconditional 0-3%.
Generator raw combined p <= 0.05 on clean windows: 9-15% (anti-conservative; the conformal step corrects it).
No redesign has been made yet; the decision is the user's.
