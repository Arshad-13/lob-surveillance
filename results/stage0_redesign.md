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
