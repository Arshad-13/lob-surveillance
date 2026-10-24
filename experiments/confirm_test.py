"""
Test-day confirmation of the repaired design V3 (rule: results/test_confirmation.md).

  run      one symbol: reference set A, first calibration day, evaluation days; saves raw per-statistic p-values
           (<out>/raw/confirm_<split>_<ds>_seed<seed>.npz) and run metadata (<out>/confirm_<split>_<ds>_seed<seed>.json).
           Prints progress only, never a detection result.
  decide   all symbols: V3 conformal p-values, measures and the outcome (<out>/confirm_<split>_decision.json)

--split test is the confirmatory run: reference = validation days 4-5, first calibration = validation day 6,
evaluation = test days T1-T6. --split val replays the same pipeline on validation data (reference days 1-2,
calibration day 3, evaluation days 4-6) and exists for smoke tests.

Usage: python -m experiments.confirm_test run --dataset BN_BTCUSDT --split test
       python -m experiments.confirm_test decide --split test
       smoke: run --split val --max-per-day 20 --max-ref 200 --m 50 --m-refine 100 --out <scratch dir>
"""

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

from experiments.followup import ALPHAS, FAMILIES, PRIMARY, SYMBOLS, arm_metrics, short
from experiments.pilot_power import fast_threshold
from experiments.pilot_strength import GRID, PREVALENCE, Q, VARIANTS, variants
from lobsurv import injection as inj
from lobsurv.confirm import bootstrap_pooled_diff, part_days, rolling_from_raw
from lobsurv.followup import auc, family_means, matched_resolution, size_matched_power
from lobsurv.null_model import GeneratorNull, generator_provenance
from lobsurv.scoring import evaluate_mixture, generator_pvalues, score, stat_matrix, uncond_pvalues
from lobsurv.statistics import LOG_DT, make_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FA_LIMIT, BOOT_REPS = 0.10, 2000
CODE_FILES = ["experiments/confirm_test.py", "experiments/pilot_strength.py", "experiments/pilot_power.py",
              "experiments/followup.py"] + sorted(os.path.relpath(p, ROOT).replace(os.sep, "/")
                                                  for p in glob.glob(os.path.join(ROOT, "lobsurv", "*.py")))


def code_sha256():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n")).hexdigest()
            for f in CODE_FILES}


def stem(split, ds, seed):
    return f"confirm_{split}_{ds}_seed{seed}"


def run(a):
    t_start = time.perf_counter()
    provenance = generator_provenance()
    rng = np.random.default_rng(a.seed)
    parts = ("train", "val", "test") if a.split == "test" else ("train", "val")
    null = GeneratorNull(a.dataset, parts=parts)
    S, view, H, G = null.stream, null.view, null.horizon, null.G
    assert G["segments"] is not None, "day boundaries are needed"
    train_phys = view.to_phys(S[:G["n_train"]])
    fast, units = fast_threshold(train_phys[:, LOG_DT]), inj.Units.from_train(train_phys)
    days = lambda part: part_days(G["offsets"], G["segments"], len(S), part, null.context, H)
    val = days("val")
    assert len(val) == 6, f"expected 6 validation days, got {len(val)}"
    if a.split == "test":
        ev = days("test")
        assert len(ev) == 6, f"expected 6 test days, got {len(ev)}"
        assert min(d.min() for d in ev) - null.context >= G["offsets"]["test"]
        ref_days, cal0 = val[3:5], val[5]
    else:
        ref_days, cal0, ev = val[0:2], val[2], val[3:]
    assert max(cal0) + H <= min(d.min() for d in ev) - null.context, "calibration must precede evaluation"

    def cap(pos, k):
        return pos if not k or len(pos) <= k else np.sort(rng.choice(pos, k, replace=False))

    A = cap(np.concatenate(ref_days), a.max_ref)
    C = cap(cal0, a.max_per_day)
    E = [cap(d, a.max_per_day) for d in ev]
    panel_at = lambda s: make_panel(view, null.has_dt, S[s - 1], fast_log_dt=fast)
    names = list(panel_at(A[0]))
    ref = np.concatenate([stat_matrix(panel_at(s), S[s:s + H][None], names) for s in A])
    floor = 2.0 / (a.m + 1)

    def score_windows(positions, injected, label):
        gf, ga, up = [], [], []
        t0 = time.perf_counter()
        for i, s in enumerate(positions):
            panel = panel_at(s)
            if injected:
                windows = view.to_norm(variants(view.to_phys(S[s:s + H]), view.to_phys(S[s - 1]), rng, units))
            else:
                windows = S[s:s + H][None]
            obs = stat_matrix(panel, windows, names)
            sim = stat_matrix(panel, null.simulate(s, a.m, rng), names)
            g1 = generator_pvalues(sim, obs, rng=rng)
            g = g1
            if a.m_refine > a.m and (g1 <= floor).any():
                sim = np.concatenate([sim, stat_matrix(panel, null.simulate(s, a.m_refine - a.m, rng), names)])
                g = generator_pvalues(sim, obs, rng=rng)
            gf.append(g1)
            ga.append(g)
            up.append(uncond_pvalues(ref, obs, rng=rng))
            if i % 200 == 0:
                print(f"[{a.dataset}] {label} {i}/{len(positions)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return np.array(gf), np.array(ga), np.array(up)

    cal_gf, cal_ga, cal_u = score_windows(C, False, "calibration")
    ev_gf, ev_ga, ev_u = [], [], []
    for k, pos in enumerate(E):
        f, g, u = score_windows(pos, True, f"eval day {k + 1}")
        ev_gf.append(f), ev_ga.append(g), ev_u.append(u)
    ev_day = np.concatenate([np.full(len(p), k) for k, p in enumerate(E)])
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "raw", stem(a.split, a.dataset, a.seed) + ".npz"),
                        m=a.m, statistics=np.array(names), A_pos=A, cal_pos=C, ev_pos=np.concatenate(E), ev_day=ev_day,
                        cal_g_first=cal_gf[:, 0], cal_g_all=cal_ga[:, 0], cal_unc=cal_u[:, 0],
                        ev_g_first=np.concatenate(ev_gf), ev_g_all=np.concatenate(ev_ga), ev_unc=np.concatenate(ev_u))
    meta = {"dataset": a.dataset, "split": a.split, "seed": a.seed, "m": a.m, "m_refine": a.m_refine,
            "statistics": names, "grid": GRID, "variants": VARIANTS, "fast_log_dt": fast,
            "n": {"A": len(A), "calibration": len(C), "eval_by_day": [len(p) for p in E]},
            "provenance": provenance, "code_sha256": code_sha256(), "seconds": time.perf_counter() - t_start}
    with open(os.path.join(out, stem(a.split, a.dataset, a.seed) + ".json"), "w") as f:
        json.dump(meta, f)
    print(f"[{a.dataset}] done in {meta['seconds'] / 3600:.2f} h", flush=True)


def v3(z):
    """V3 conformal p-values {det: [n, variants]} and V3 scores, from one symbol's raw file."""
    m = int(z["m"])
    g_cal = matched_resolution(z["cal_g_first"], z["cal_g_all"], m)
    g_ev = matched_resolution(z["ev_g_first"], z["ev_g_all"], m)
    day = z["ev_day"]
    p = {"gen": rolling_from_raw(g_cal, g_ev, day), "unc": rolling_from_raw(z["cal_unc"], z["ev_unc"], day)}
    sc = {"gen": (score(g_cal), score(g_ev)), "unc": (score(z["cal_unc"]), score(z["ev_unc"]))}
    return p, sc


def bh_by_day(cal_sc, ev_sc, day, seed):
    """BH at q = Q on 5%-prevalence mixtures within each day (calibration as in V3); mean over days per variant."""
    days = sorted(int(d) for d in np.unique(day))
    out = {}
    for v, name in enumerate(VARIANTS[1:], start=1):
        rec = []
        for k, d in enumerate(days):
            cal = cal_sc if k == 0 else ev_sc[day == days[k - 1], 0]
            rows = day == d
            rec.append(evaluate_mixture(cal, ev_sc[rows, 0], ev_sc[rows, v], Q, PREVALENCE,
                                        np.random.default_rng(seed + 100 * k + v), reps=50))
        out[name] = {key: float(np.mean([r[key] for r in rec])) for key in rec[0]}
    return out


def decide(a):
    out_dir = os.path.abspath(a.out)
    symbols = a.symbols or SYMBOLS
    res = {"split": a.split, "symbols": symbols, "integrity": {}, "per_symbol": {}}
    stage0 = {}
    for ds in symbols:
        with open(os.path.join(ROOT, "results", f"pilot_strength_{ds}_seed0.json")) as f:
            stage0[ds] = json.load(f)["provenance"]["file_sha256"]
    diffs = {al: {f: [] for f in FAMILIES} for al in ALPHAS}
    for ds in symbols:
        with open(os.path.join(out_dir, stem(a.split, ds, a.seed) + ".json")) as f:
            meta = json.load(f)
        res["integrity"][ds] = {"generator_files_match_stage0": meta["provenance"]["file_sha256"] == stage0[ds],
                                "code_sha256": meta["code_sha256"]}
        z = np.load(os.path.join(out_dir, "raw", stem(a.split, ds, a.seed) + ".npz"))
        p, sc = v3(z)
        day = z["ev_day"]
        rec = {det: arm_metrics(p[det], VARIANTS) for det in ("gen", "unc")}
        for det in ("gen", "unc"):
            r = rec[det]
            r["false_alarm_by_day"] = {int(d) + 1: float(np.mean(p[det][day == d, 0] <= PRIMARY))
                                       for d in np.unique(day)}
            for al in ALPHAS:
                pw = {v: size_matched_power(p[det][:, 0], p[det][:, i], al) for i, v in enumerate(VARIANTS) if i}
                r[f"matched_power_{al}"] = family_means(pw, VARIANTS)
            r["auc"] = family_means({v: auc(p[det][:, 0], p[det][:, i]) for i, v in enumerate(VARIANTS) if i},
                                    VARIANTS)
            bh = bh_by_day(*sc[det], day, a.seed)
            r["bh"] = family_means({k: v["power"] for k, v in bh.items()}, VARIANTS)
            r["bh_fdr"] = family_means({k: v["fdr"] for k, v in bh.items()}, VARIANTS)
        floor = 2.0 / (int(z["m"]) + 1)
        hit = (z["ev_g_first"] <= floor).any(-1)
        rec["refined_share"] = {"calibration": float(np.mean((z["cal_g_first"] <= floor).any(-1))),
                                "eval_clean": float(np.mean(hit[:, 0])), "eval_injected": float(np.mean(hit[:, 1:]))}
        rec["n"], rec["hours"] = meta["n"], meta["seconds"] / 3600
        res["per_symbol"][ds] = rec
        for al in ALPHAS:
            hits = {det: (p[det][:, 1:] <= al).astype(float) for det in ("gen", "unc")}
            for f in FAMILIES:
                cols = [i for i, v in enumerate(VARIANTS[1:]) if v.startswith(f + ":")]
                diffs[al][f].append(hits["gen"][:, cols].mean(1) - hits["unc"][:, cols].mean(1))

    ps = res["per_symbol"]
    res["integrity_ok"] = all(v["generator_files_match_stage0"] for v in res["integrity"].values())
    two_se = {ds: 0.05 + 2 * np.sqrt(0.05 * 0.95 / ps[ds]["gen"]["n"]) for ds in symbols}
    res["c1"] = all(ps[ds][det]["false_alarm"][str(PRIMARY)] <= FA_LIMIT for ds in symbols for det in ("gen", "unc"))
    res["within_nominal_2se"] = {det: [short(ds) for ds in symbols
                                       if ps[ds][det]["false_alarm"][str(PRIMARY)] > two_se[ds]]
                                 for det in ("gen", "unc")}
    res["pooled"] = {}
    for key in [f"power_{al}" for al in ALPHAS] + [f"matched_power_{al}" for al in ALPHAS] + ["auc", "bh", "bh_fdr"]:
        res["pooled"][key] = {det: {f: float(np.mean([ps[ds][det][key][f] for ds in symbols])) for f in FAMILIES}
                              for det in ("gen", "unc")}
    res["power_difference"] = {}
    for al in ALPHAS:
        res["power_difference"][str(al)] = {}
        for f in FAMILIES:
            est, lo, hi = bootstrap_pooled_diff(diffs[al][f], reps=BOOT_REPS, seed=a.seed)
            res["power_difference"][str(al)][f] = {"estimate": est, "lo": lo, "hi": hi, "confirmed": bool(lo > 0),
                                                   "symbols_gen_higher": int(sum(d.mean() > 0 for d in diffs[al][f]))}
    confirmed = [f for f in FAMILIES if res["power_difference"][str(PRIMARY)][f]["confirmed"]]
    res["c2"] = len(confirmed) >= 2
    res["confirmed_families"] = confirmed
    if not res["integrity_ok"]:
        res["outcome"] = "VOID (generator files differ from Stage 0)"
    elif not res["c1"]:
        res["outcome"] = "NOT CONFIRMED (validity)"
    elif not res["c2"]:
        res["outcome"] = "NOT CONFIRMED (power)"
    else:
        res["outcome"] = "CONFIRMED"
    with open(os.path.join(out_dir, f"confirm_{a.split}_decision.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("generator files match Stage 0:", res["integrity_ok"])
    for det in ("gen", "unc"):
        print(det, "false alarm @0.05:", {short(ds): round(ps[ds][det]["false_alarm"][str(PRIMARY)], 3)
                                          for ds in symbols})
    for f in FAMILIES:
        d = res["power_difference"][str(PRIMARY)][f]
        print(f"{f:9s} power gen {res['pooled']['power_0.05']['gen'][f]:.3f} unc "
              f"{res['pooled']['power_0.05']['unc'][f]:.3f}  diff {d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]")
    print("C1", res["c1"], "| C2", res["c2"], confirmed, "| outcome:", res["outcome"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "decide"])
    ap.add_argument("--dataset", default="BN_BTCUSDT")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--symbols", nargs="*", help="decide: symbols to include (default: all six)")
    ap.add_argument("--max-per-day", type=int, default=0, help="cap on windows per day (0: every window)")
    ap.add_argument("--max-ref", type=int, default=1500)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--m-refine", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    {"run": run, "decide": decide}[args.step](args)
