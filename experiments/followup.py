"""
Stage-0 follow-up (rule: results/stage0_followup.md). Validation data only.

  q1   size-matched power and AUC from the saved Stage-0 files (results/pilot_strength_<ds>_seed0.json)
  q2   integrity check and arms V1-V3 from the rerun (results/raw/pilot_strength_<ds>_seed0.npz and
       results/pilot_strength_<ds>_seed0_rerun.json), then the overall outcome

Usage: python -m experiments.followup q1 | q2
"""

import argparse
import json
import os

import numpy as np

from lobsurv.followup import (auc, conformal_from_raw, family_means, matched_resolution, rolling_conformal,
                              size_matched_power)
from lobsurv.scoring import score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SYMBOLS = ["BN_BTCUSDT", "BN_ETHUSDT", "BN_SOLUSDT", "BN_XRPUSDT", "BN_DOGEUSDT", "BN_LTCUSDT"]
FAMILIES = ["spoof", "stuffing", "ignition"]
ALPHAS, PRIMARY, FA_LIMIT = (0.01, 0.05), 0.05, 0.10


def load(ds, suffix=""):
    with open(os.path.join(RES, f"pilot_strength_{ds}_seed0{suffix}.json")) as f:
        return json.load(f)


def short(ds):
    return ds.replace("BN_", "").replace("USDT", "")


def q1():
    out = {"alphas": ALPHAS, "per_symbol": {}, "pooled": {}}
    for ds in SYMBOLS:
        d = load(ds)
        rec = {}
        for det in ("gen", "unc"):
            p = np.asarray(d["detectors"][det]["conformal_p"])
            r = {"realised_false_alarm": {str(a): float(np.mean(p[:, 0] <= a)) for a in ALPHAS}}
            for a in ALPHAS:
                pw = {v: size_matched_power(p[:, 0], p[:, i], a) for i, v in enumerate(d["variants"]) if i}
                r[f"matched_power_{a}"] = family_means(pw, d["variants"])
            au = {v: auc(p[:, 0], p[:, i]) for i, v in enumerate(d["variants"]) if i}
            r["auc"] = family_means(au, d["variants"])
            r["auc_by_variant"] = au
            rec[det] = r
        out["per_symbol"][ds] = rec
    for a in ALPHAS:
        key = f"matched_power_{a}"
        out["pooled"][key] = {det: {f: float(np.mean([out["per_symbol"][ds][det][key][f] for ds in SYMBOLS]))
                                    for f in FAMILIES} for det in ("gen", "unc")}
    out["pooled"]["auc"] = {det: {f: float(np.mean([out["per_symbol"][ds][det]["auc"][f] for ds in SYMBOLS]))
                                  for f in FAMILIES} for det in ("gen", "unc")}
    key = f"matched_power_{PRIMARY}"
    wins = {}
    for f in FAMILIES:
        pooled = out["pooled"][key]["gen"][f] > out["pooled"][key]["unc"][f]
        n_sym = sum(out["per_symbol"][ds]["gen"][key][f] >= out["per_symbol"][ds]["unc"][key][f] for ds in SYMBOLS)
        wins[f] = {"pooled_higher": bool(pooled), "symbols_at_least_as_high": int(n_sym),
                   "counts": bool(pooled and n_sym >= 4)}
    out["family_wins"] = wins
    out["q1_holds"] = sum(w["counts"] for w in wins.values()) >= 2
    with open(os.path.join(RES, "followup_q1.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"Size-matched power at alpha = {PRIMARY} (gen / unc), pooled over symbols and strength grid")
    for f in FAMILIES:
        g, u = out["pooled"][key]["gen"][f], out["pooled"][key]["unc"][f]
        per = ", ".join(f"{short(ds)} {out['per_symbol'][ds]['gen'][key][f]:.2f}/"
                        f"{out['per_symbol'][ds]['unc'][key][f]:.2f}" for ds in SYMBOLS)
        print(f"  {f:9s} {g:.3f} / {u:.3f}   symbols gen >= unc: {wins[f]['symbols_at_least_as_high']}/6   ({per})")
    k1 = f"matched_power_{ALPHAS[0]}"
    print(f"alpha = {ALPHAS[0]}: " + ", ".join(f"{f} {out['pooled'][k1]['gen'][f]:.3f}/{out['pooled'][k1]['unc'][f]:.3f}"
                                         for f in FAMILIES))
    print("AUC: " + ", ".join(f"{f} {out['pooled']['auc']['gen'][f]:.3f}/{out['pooled']['auc']['unc'][f]:.3f}"
                              for f in FAMILIES))
    print("Q1 holds:", out["q1_holds"])


def arm_metrics(p, variants):
    r = {"false_alarm": {str(a): float(np.mean(p[:, 0] <= a)) for a in ALPHAS}, "n": int(len(p))}
    for a in ALPHAS:
        pw = {v: float(np.mean(p[:, i] <= a)) for i, v in enumerate(variants) if i}
        r[f"power_{a}"] = family_means(pw, variants)
    return r


def q2():
    out = {"per_symbol": {}, "integrity": {}}
    for ds in SYMBOLS:
        old, new = load(ds), load(ds, "_rerun")
        z = np.load(os.path.join(RES, "raw", f"pilot_strength_{ds}_seed0.npz"))
        m, variants, day = int(z["m"]), new["variants"], z["E_day"]
        g_orig = {"B": z["B_g_all"], "E": z["E_g_all"]}
        p_orig = {"gen": conformal_from_raw(g_orig["B"], g_orig["E"]), "unc": conformal_from_raw(z["B_unc"], z["E_unc"])}
        diff = max(float(np.max(np.abs(np.round(p_orig[det], 5) - np.asarray(old["detectors"][det]["conformal_p"]))))
                   for det in ("gen", "unc"))
        out["integrity"][ds] = diff
        if diff > 1e-5:
            print(f"{ds}: integrity check FAILED (max diff {diff:.2e}); stopping")
            out["integrity_ok"] = False
            with open(os.path.join(RES, "followup_q2.json"), "w") as f:
                json.dump(out, f, indent=1)
            return
        g_v1 = {"B": matched_resolution(z["B_g_first"], z["B_g_all"], m),
                "E": matched_resolution(z["E_g_first"], z["E_g_all"], m)}
        days = sorted(np.unique(day))

        def rolling(cal_p, eval_p):
            cal0, sc = score(cal_p), score(eval_p)
            cal = {int(days[0]): cal0}
            for prev, d in zip(days[:-1], days[1:]):
                cal[int(d)] = sc[day == prev, 0]
            return rolling_conformal(cal, sc, day), {int(d): int(len(c)) for d, c in cal.items()}

        arms = {"original": {"gen": p_orig["gen"], "unc": p_orig["unc"]},
                "V1": {"gen": conformal_from_raw(g_v1["B"], g_v1["E"]), "unc": p_orig["unc"]}}
        arms["V2"] = {"gen": rolling(g_orig["B"], g_orig["E"])[0], "unc": rolling(z["B_unc"], z["E_unc"])[0]}
        arms["V3"] = {"gen": rolling(g_v1["B"], g_v1["E"])[0], "unc": arms["V2"]["unc"]}
        n_cal = rolling(z["B_unc"], z["E_unc"])[1]
        rec = {arm: {det: arm_metrics(p, variants) for det, p in v.items()} for arm, v in arms.items()}
        rec["n_cal_rolling"] = n_cal
        rec["false_alarm_by_day"] = {arm: {det: {int(d): float(np.mean(p[day == d, 0] <= PRIMARY)) for d in days}
                                           for det, p in v.items()} for arm, v in arms.items()}
        floor = 2.0 / (m + 1)
        rec["refined_share"] = {"B": float(np.mean((z["B_g_first"] <= floor).any(-1))),
                                "E_clean_original": float(np.mean((z["E_g_first"] <= floor).any(-1).any(-1))),
                                "E_clean_V1": float(np.mean((z["E_g_first"][:, 0] <= floor).any(-1)))}
        out["per_symbol"][ds] = rec
    out["integrity_ok"] = True

    def fa_ok(arm, limit):
        return all(out["per_symbol"][ds][arm][det]["false_alarm"][str(PRIMARY)] <= limit
                   for ds in SYMBOLS for det in ("gen", "unc"))

    se = {arm: max(0.05 + 2 * np.sqrt(0.05 * 0.95 / out["per_symbol"][ds][arm]["gen"]["n"]) for ds in SYMBOLS)
          for arm in ("original", "V1", "V2", "V3")}
    out["within_limit"] = {arm: fa_ok(arm, FA_LIMIT) for arm in ("original", "V1", "V2", "V3")}
    out["within_nominal_2se"] = {arm: fa_ok(arm, se[arm]) for arm in ("original", "V1", "V2", "V3")}
    repaired = next((arm for arm in ("V1", "V3") if out["within_limit"][arm]), None)
    out["repaired_design"] = repaired
    key = f"power_{PRIMARY}"
    if repaired:
        pooled = {det: {f: float(np.mean([out["per_symbol"][ds][repaired][det][key][f] for ds in SYMBOLS]))
                        for f in FAMILIES} for det in ("gen", "unc")}
        out["repaired_pooled_power"] = pooled
        out["repaired_power_wins"] = sum(pooled["gen"][f] > pooled["unc"][f] for f in FAMILIES)
    with open(os.path.join(RES, "followup_q1.json")) as f:
        q1_holds = json.load(f)["q1_holds"]
    if not q1_holds:
        out["outcome"] = "negative"
    elif repaired and out["repaired_power_wins"] >= 2:
        out["outcome"] = "GO"
    else:
        out["outcome"] = "mixed"
    with open(os.path.join(RES, "followup_q2.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("integrity max diff:", {short(ds): f"{v:.1e}" for ds, v in out["integrity"].items()})
    for arm in ("original", "V1", "V2", "V3"):
        fa = ", ".join(f"{short(ds)} {out['per_symbol'][ds][arm]['gen']['false_alarm'][str(PRIMARY)]:.3f}/"
                       f"{out['per_symbol'][ds][arm]['unc']['false_alarm'][str(PRIMARY)]:.3f}" for ds in SYMBOLS)
        pw = ", ".join(f"{f} {np.mean([out['per_symbol'][ds][arm]['gen'][key][f] for ds in SYMBOLS]):.3f}/"
                       f"{np.mean([out['per_symbol'][ds][arm]['unc'][key][f] for ds in SYMBOLS]):.3f}" for f in FAMILIES)
        print(f"{arm:8s} false alarm gen/unc: {fa}\n         power: {pw}   within 0.10: {out['within_limit'][arm]}")
    for arm in ("original", "V3"):
        print(f"false alarm by day ({arm}, gen):", {short(ds): {d: round(v, 3) for d, v in out["per_symbol"][ds][
            "false_alarm_by_day"][arm]["gen"].items()} for ds in SYMBOLS})
    print("refined share:", {short(ds): out["per_symbol"][ds]["refined_share"] for ds in SYMBOLS})
    print("repaired design:", repaired, "| outcome:", out["outcome"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["q1", "q2"])
    {"q1": q1, "q2": q2}[ap.parse_args().part]()
