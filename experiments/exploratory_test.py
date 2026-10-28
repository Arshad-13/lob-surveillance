"""
Exploratory analyses of the test-day run, added after its decision (results/test_confirmation.md). No rule governs
them and they do not change the outcome.

  intervals   the power differences of the decision rule with intervals that treat symbols as random: a t interval
              over the six per-symbol means and a two-stage bootstrap (symbols, then windows within each symbol);
              the same for size-matched power and AUC (t interval only, one value per symbol)
  days        what was different on the days with large unconditional false-alarm rates: activity descriptors of
              each test day and of its calibration day, and the share of clean windows with a per-statistic
              p-value <= 0.01 under each null

Reads results/raw/confirm_test_*.npz and the real streams; writes results/exploratory_test.json.

Usage: python -m experiments.exploratory_test
"""

import json
import os

import numpy as np
from scipy import stats

from experiments.confirm_test import ROOT, stem, v3
from experiments.followup import FAMILIES, PRIMARY, SYMBOLS, short
from experiments.pilot_strength import VARIANTS
from lobsurv.followup import auc, matched_resolution, size_matched_power
from lobsurv.null_model import GeneratorNull
from lobsurv.statistics import LOG_DT

REPS, SEED, TAIL = 2000, 0, 0.01


def family_cols(f):
    return [i for i, v in enumerate(VARIANTS) if i and v.startswith(f + ":")]


def t_interval(x, level=0.95):
    x = np.asarray(x, float)
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def two_stage(diffs, rng, level=0.95):
    """Mean of per-symbol window means; resample symbols, then windows within each drawn symbol."""
    k = len(diffs)
    boot = np.empty(REPS)
    for r in range(REPS):
        pick = rng.integers(0, k, k)
        boot[r] = np.mean([diffs[j][rng.integers(0, len(diffs[j]), len(diffs[j]))].mean() for j in pick])
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def intervals(raw):
    rng = np.random.default_rng(SEED)
    out = {}
    for f in FAMILIES:
        cols = family_cols(f)
        diffs, per = [], {"power": [], "matched_power": [], "auc": []}
        for ds in SYMBOLS:
            p = raw[ds]["p"]
            hit = {d: (p[d][:, cols] <= PRIMARY).mean(1) for d in ("gen", "unc")}
            diffs.append(hit["gen"] - hit["unc"])
            per["power"].append(float(diffs[-1].mean()))
            mp = {d: np.mean([size_matched_power(p[d][:, 0], p[d][:, i], PRIMARY) for i in cols]) for d in p}
            per["matched_power"].append(float(mp["gen"] - mp["unc"]))
            au = {d: np.mean([auc(p[d][:, 0], p[d][:, i]) for i in cols]) for d in p}
            per["auc"].append(float(au["gen"] - au["unc"]))
        est, lo, hi = t_interval(per["power"])
        blo, bhi = two_stage(diffs, rng)
        out[f] = {"power": {"estimate": est, "t_lo": lo, "t_hi": hi, "boot_lo": blo, "boot_hi": bhi,
                            "per_symbol": dict(zip(map(short, SYMBOLS), per["power"]))}}
        for key in ("matched_power", "auc"):
            est, lo, hi = t_interval(per[key])
            out[f][key] = {"estimate": est, "t_lo": lo, "t_hi": hi, "per_symbol": dict(zip(map(short, SYMBOLS), per[key]))}
    return out


def describe(view, S, positions, H):
    """Activity of the windows at `positions`: medians and shares over all their events, physical units."""
    X = view.to_phys(np.concatenate([S[s:s + H] for s in positions]))
    return {"move_share": float(np.mean(X[:, 0] != 0)), "abs_move": float(np.mean(np.abs(X[:, 0]))),
            "spread": float(np.mean(X[:, 1])), "log_queue": float(np.mean(X[:, 2:4])),
            "ms_per_event": float(1000 * np.mean(np.exp(X[:, LOG_DT])))}


def jump_correlation(rec):
    """Spearman correlation, over symbol-days, of each detector's false-alarm rate with the log ratio of the day's
    mid-move share to that of its calibration day (the day before)."""
    jump, fa = [], {"gen": [], "unc": []}
    for days_ in rec.values():
        keys = list(days_)
        for prev, d in zip(keys[:-1], keys[1:]):
            jump.append(np.log(days_[d]["activity"]["move_share"] / days_[prev]["activity"]["move_share"]))
            for k in fa:
                fa[k].append(days_[d]["false_alarm"][k])
    out = {"n_days": len(jump)}
    for k in fa:
        rho, p = stats.spearmanr(jump, fa[k])
        out[k] = {"rho": float(rho), "p": float(p)}
    return out


def days(raw):
    out = {}
    for ds in SYMBOLS:
        z, p = raw[ds]["z"], raw[ds]["p"]
        null = GeneratorNull(ds, parts=("train", "val", "test"))
        H, day = null.horizon, z["ev_day"]
        m = int(z["m"])
        stat_p = {"gen": (matched_resolution(z["cal_g_first"], z["cal_g_all"], m),
                          matched_resolution(z["ev_g_first"], z["ev_g_all"], m)[:, 0]),
                  "unc": (z["cal_unc"], z["ev_unc"][:, 0])}
        names = [str(s) for s in z["statistics"]]
        rec = {"calibration": {"activity": describe(null.view, null.stream, z["cal_pos"], H),
                               "tail_share": {d: dict(zip(names, (stat_p[d][0] <= TAIL).mean(0).tolist()))
                                              for d in stat_p}}}
        for d in sorted(int(x) for x in np.unique(day)):
            rows = day == d
            rec[f"T{d + 1}"] = {"activity": describe(null.view, null.stream, z["ev_pos"][rows], H),
                                "false_alarm": {k: float(np.mean(p[k][rows, 0] <= PRIMARY)) for k in p},
                                "tail_share": {k: dict(zip(names, (stat_p[k][1][rows] <= TAIL).mean(0).tolist()))
                                               for k in stat_p}}
        out[short(ds)] = rec
        print(f"[{short(ds)}] days done", flush=True)
    return out


def main():
    raw = {}
    for ds in SYMBOLS:
        z = np.load(os.path.join(ROOT, "results", "raw", stem("test", ds, 0) + ".npz"))
        raw[ds] = {"z": z, "p": v3(z)[0]}
    res = {"note": "exploratory, after the test-day decision; no rule governs these numbers",
           "alpha": PRIMARY, "tail": TAIL, "intervals": intervals(raw), "days": days(raw)}
    res["jump_correlation"] = jump_correlation(res["days"])
    print("false alarm vs activity jump:", res["jump_correlation"])
    with open(os.path.join(ROOT, "results", "exploratory_test.json"), "w") as f:
        json.dump(res, f, indent=1)
    for f_ in FAMILIES:
        r = res["intervals"][f_]["power"]
        print(f"{f_:9s} diff {r['estimate']:+.3f}  t [{r['t_lo']:+.3f}, {r['t_hi']:+.3f}]  "
              f"boot [{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}]")


if __name__ == "__main__":
    main()
