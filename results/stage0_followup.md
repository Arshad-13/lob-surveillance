# Stage-0 follow-up (written 2026-10-24, before any new analysis of the Stage-0 files)

The Stage-0 redesign came out NO-GO (`stage0_redesign.md`): the generator-conditional detector had higher power
on all three families, but its realised false-alarm rate on clean evaluation windows was 0.110 on XRP (limit
0.10) and above the nominal 0.05 on four of six symbols. That verdict stands and is reported as found. This
follow-up asks two separate questions about it. It uses validation data only; the test split stays unread.

What had been looked at when this was written: the per-family power averages, the per-symbol false-alarm rates,
and the run metadata (`n`, `n_refined`, `seconds`). The per-window conformal p-values had not been analysed.

## Two candidate causes of the false-alarm excess

1. **Resolution asymmetry (found in the metadata while writing this rule).** A window gets 1,000
   continuations instead of 200 when any of its variants has a statistic with Monte Carlo p <= 2/201. In the
   evaluation set E the strong injections almost always trigger this, so the clean copy of 96-100% of E windows
   was scored with 1,000 continuations; in the calibration set B only 7-13% of windows were. The Monte Carlo
   p-value (1 + #)/(M + 1) is more conservative at M = 200, so calibration scores sit lower than clean
   evaluation scores. The rule in `stage0_redesign.md` meant the refinement to be applied alike to B and E; it
   was not, because in E it was triggered by a window's injected siblings.
2. **Calibration-day drift.** B is validation day 3; E is days 4-6.

## Q1: does the power advantage survive at matched size? (saved files, no new run)

For each symbol and detector, set the threshold t(alpha) to the largest conformal p-value whose share among the
clean E windows is at most alpha, and compute size-matched power = share of injected windows with p <= t(alpha).
alpha = 0.05 is primary, 0.01 secondary. Aggregation as in the original rule: average over the strength grid
of each family and over the six symbols. The threshold is set on the evaluation windows themselves, so this is
a diagnostic of ranking quality, not a deployable procedure. Also reported, not part of the decision: the
paired AUC per variant (clean vs injected copy of the same windows, ties count one half) and per-symbol wins.

**Q1 holds** if the generator-conditional detector has higher size-matched power at alpha = 0.05 on at least
2 of the 3 families, and on each such family is at least as high on at least 4 of the 6 symbols.

## Q2: can validity be repaired, and which cause was it? (one rerun)

Rerun `experiments/pilot_strength.py` with the same seed and settings, saving, for every window and variant,
the generator's per-statistic p-values from the first 200 continuations and from all continuations, the
unconditional p-values, and the window's validation day. The added saving must not change any computation.

**Integrity check first.** The rerun's conformal p-values under the original rule must equal the saved ones
(max absolute difference <= 1e-5) on every symbol. If not, stop and report; no arm below is computed.

Arms (generator-conditional; the unconditional detector is recomputed with the same calibration sets):

- **V1, matched resolution:** each window-variant's generator p-values come from 200 continuations, or from
  1,000 only if that variant itself has a statistic with p <= 2/201 at 200, which is the rule B already used.
  Calibration as before (day 3).
- **V2, rolling calibration:** evaluation day d is calibrated on the clean windows of day d-1 (day 4 on B; days
  5 and 6 on the clean E windows of days 4 and 5), original resolution.
- **V3, both.**

**Decision.** The repaired design is the first of V1, V3 (simplest first) under which both detectors have a
realised false-alarm rate at alpha = 0.05 of at most 0.10 on every symbol. V2 is reported for attribution
only. Within-nominal is reported as well: false-alarm rate <= 0.05 + 2 SE (0.061 at 1,500 windows; computed
from each arm's actual window count).

**Attribution (descriptive, no decision):** false-alarm rate by evaluation day under the original rule (drift
predicts a rise with distance from day 3); the change from the original to V1 (resolution) and from V1 to V3
(drift).

## Overall outcome

- **GO to the main study design:** Q1 holds, a repaired design exists, and under it the generator-conditional
  detector has higher power at alpha = 0.05 (not size-matched) on at least 2 of 3 families. The repaired design
  replaces the original in every later stage, and the Stage-0 NO-GO is disclosed.
- **Mixed:** Q1 holds but no repaired design meets the limit. The paper reports a ranking advantage without a
  validated false-alarm control.
- **Negative:** Q1 fails. The paper reports that the power advantage came from a looser effective threshold.

Whatever the outcome, the paper reports the original NO-GO, Q1, every arm of Q2 and the integrity check.

## Checks done before running

- The size-matching, AUC and arm computations are tested on synthetic inputs with ties and discrete p-values
  (`tests/test_followup.py`), not on the Stage-0 files.
- Treatment steps: under V1, the share of refined clean E windows must fall to the share of refined B windows'
  order (reported); under V2/V3, the calibration day is the day before by construction.
- Tie-break for t(alpha): the largest observed p-value whose clean share is <= alpha; if no clean window has
  p <= alpha, t(alpha) = 0 and power is 0.
