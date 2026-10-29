"""
Injection seeds on the validation days, added after the test-day decision (exploratory; no rule governs it). Seeds 1
and 2 rerun Stage 0 with --save-raw (jobs 32842, 32843); seed 0 is the follow-up rerun (job 32720). Each seed is
scored with arm V3 of the follow-up (matched refinement, each day calibrated on the day before), the design used on
the test days.

Reads results/raw/pilot_strength_<ds>_seed<k>.npz and results/pilot_strength_<ds>_seed<k>_rerun.json; writes
results/seeds_validation.json.

Usage: python -m experiments.seeds
"""

import json
import os

import numpy as np

from experiments.followup import FAMILIES, PRIMARY, RES, SYMBOLS, short
from lobsurv.followup import (auc, conformal_from_raw, family_means, matched_resolution, rolling_conformal,
                              size_matched_power)
from lobsurv.scoring import score

SEEDS = (0, 1, 2)


def v3(z):
    """V3 conformal p-values [n_E, variants] of both detectors, as in experiments.followup.q2."""
    m, day = int(z["m"]), z["E_day"]
    days = sorted(np.unique(day))

    def rolling(cal_p, eval_p):
        sc = score(eval_p)
        cal = {int(days[0]): score(cal_p)}
        for prev, d in zip(days[:-1], days[1:]):
            cal[int(d)] = sc[day == prev, 0]
        return rolling_conformal(cal, sc, day)

    gen = rolling(matched_resolution(z["B_g_first"], z["B_g_all"], m), matched_resolution(z["E_g_first"], z["E_g_all"], m))
    return {"gen": gen, "unc": rolling(z["B_unc"], z["E_unc"])}


def main():
    out = {"note": "exploratory, after the test-day decision", "alpha": PRIMARY, "integrity": {}, "per_seed": {}}
    for k in SEEDS:
        rec = {}
        for ds in SYMBOLS:
            z = np.load(os.path.join(RES, "raw", f"pilot_strength_{ds}_seed{k}.npz"))
            with open(os.path.join(RES, f"pilot_strength_{ds}_seed{k}_rerun.json")) as f:
                js = json.load(f)
            variants = js["variants"]
            orig = {"gen": conformal_from_raw(z["B_g_all"], z["E_g_all"]), "unc": conformal_from_raw(z["B_unc"], z["E_unc"])}
            out["integrity"][f"{short(ds)}_seed{k}"] = max(
                float(np.max(np.abs(np.round(orig[d], 5) - np.asarray(js["detectors"][d]["conformal_p"])))) for d in orig)
            p = v3(z)
            r = {}
            for d in p:
                cols = range(1, len(variants))
                r[d] = {"false_alarm": float(np.mean(p[d][:, 0] <= PRIMARY)),
                        "power": family_means({variants[i]: float(np.mean(p[d][:, i] <= PRIMARY)) for i in cols}, variants),
                        "matched_power": family_means({variants[i]: size_matched_power(p[d][:, 0], p[d][:, i], PRIMARY)
                                                       for i in cols}, variants),
                        "auc": family_means({variants[i]: auc(p[d][:, 0], p[d][:, i]) for i in cols}, variants)}
            rec[short(ds)] = r
        pooled = {d: {key: {f: float(np.mean([rec[s][d][key][f] for s in rec])) for f in FAMILIES}
                      for key in ("power", "matched_power", "auc")} for d in ("gen", "unc")}
        for d in pooled:
            pooled[d]["max_false_alarm"] = max(rec[s][d]["false_alarm"] for s in rec)
        out["per_seed"][str(k)] = {"per_symbol": rec, "pooled": pooled}
    with open(os.path.join(RES, "seeds_validation.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("integrity (max |diff| to saved conformal p):", max(out["integrity"].values()))
    for k in SEEDS:
        P = out["per_seed"][str(k)]
        fa = {d: " ".join(f"{s} {P['per_symbol'][s][d]['false_alarm']:.3f}" for s in P["per_symbol"]) for d in ("gen", "unc")}
        print(f"seed {k}: FA gen [{fa['gen']}]\n        FA unc [{fa['unc']}]")
        for key in ("power", "matched_power", "auc"):
            print(f"  {key:13s} " + "  ".join(f"{f} {P['pooled']['gen'][key][f]:.3f}/{P['pooled']['unc'][key][f]:.3f}"
                                              for f in FAMILIES))
        for f in FAMILIES:
            n = sum(P["per_symbol"][s]["gen"]["matched_power"][f] > P["per_symbol"][s]["unc"]["matched_power"][f]
                    for s in P["per_symbol"])
            print(f"    {f}: matched power gen > unc on {n}/6 symbols")


if __name__ == "__main__":
    main()
