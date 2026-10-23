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
