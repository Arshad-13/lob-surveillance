# Stage-0 redesign (written 2026-10-23, before the full redesigned run)

Why: the first pilot's rule (BH power at 5% prevalence) could not register any discovery at its calibration
size (results/stage0_pilot_plan.md). Approved by the user on 2026-10-23.

Changes (validation split only; the test split is still unread):
1. Primary measure: power at a fixed false-alarm rate, i.e. the share of injected windows with conformal
   p <= alpha, at alpha = 0.05 (primary) and 0.01 (secondary), across a grid of strengths:
   spoof k in {5, 10, 20, 50, 100} (tau = 20); stuffing n in {10, 25, 50, 100}; ignition m in {1, 2, 3, 5}
   (r = 10). Strengths are paired: one position, side and direction per family and window.
   BH power and FDR at q = 0.10 and 5% prevalence are reported as secondary measures.
2. New statistic queue_roundtrip (a same-price jump up in a best queue followed within 50 events by a jump down).
3. Resolution parity: the generator null uses 200 continuations, raised to 1,000 for any window (clean or
   injected, calibration or evaluation) in which a statistic beats nearly all of them (p <= 2/201). The
   unconditional null's reference set has about 1,500 windows.
4. Sizes per symbol: reference A = up to 1,500 windows of validation days 1-2; calibration B = all windows of
   day 3 (about 1,000); evaluation E = up to 1,500 windows of days 4-6. Six Binance symbols, seed 0.
Smoke tests behind these choices: BN_LTCUSDT validation, 150 windows per set, 50 (then 300) continuations.

Rule: for each family, average the power at alpha = 0.05 over its strength grid and over the six symbols.
GO if the generator-conditional detector beats the unconditional one on at least 2 of the 3 families, and
both detectors' realised false-alarm rate on clean evaluation windows is at most 2 alpha (0.10) on every symbol.
Otherwise NO-GO: the comparison is reported as found, and the paper becomes a documented negative or
mixed result.

## Result (2026-10-24, cluster job 32705; read after all six symbols finished)

Power at alpha = 0.05, averaged over the strength grid and the six symbols (generator-conditional / unconditional):

| Family | gen | unc | Per symbol (gen / unc) |
|---|---|---|---|
| spoof | 0.691 | 0.414 | BTC 0.70/0.33, DOGE 0.66/0.30, ETH 0.54/0.21, LTC 0.76/0.44, SOL 0.60/0.51, XRP 0.88/0.70 |
| stuffing | 0.275 | 0.163 | BTC 0.19/0.10, DOGE 0.31/0.12, ETH 0.21/0.12, LTC 0.40/0.24, SOL 0.29/0.26, XRP 0.24/0.14 |
| ignition | 0.419 | 0.219 | BTC 0.24/0.05, DOGE 0.58/0.11, ETH 0.19/0.04, LTC 0.68/0.45, SOL 0.08/0.10, XRP 0.74/0.57 |

Realised false-alarm rate on clean evaluation windows at alpha = 0.05 (gen / unc): BTC 0.051/0.027,
DOGE 0.072/0.026, ETH 0.047/0.025, LTC 0.082/0.041, SOL 0.059/0.079, XRP **0.110**/0.035.

Verdict under the rule: **NO-GO.** The generator-conditional detector has higher power on all 3 families, but
its false-alarm rate on XRP (0.110) exceeds the 0.10 limit. Its false-alarm rate is also above the nominal 0.05
on four of six symbols (unconditional: one of six), so part of its power advantage may come from a looser
effective threshold. The calibration windows (day 3) and the evaluation windows (days 4-6) differ in time;
drift between them is the likely cause, but this has not been tested. Per the rule, the comparison is reported
as found and the rule is not changed after the fact.
