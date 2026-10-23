"""
Stage-0 power pilot (proposal Section 7): on validation data only, does the generator-conditional null detect
injected manipulation footprints better than the unconditional null built from the same statistics?

Validation days of one symbol are split by day into
  A  reference windows for the unconditional null (first two days)
  B  clean calibration windows for the conformal step, shared by both detectors (third day)
  E  evaluation windows (remaining days): each scored clean and with one injection per family
Windows are non-overlapping 100-event stretches, each with the 100 real events before it as context; injections
start 10-60 events into the window. Reported per family, at the proposal's 5% prevalence and BH level q = 0.10:
power and false-discovery proportion of both detectors, plus the share of injected windows with conformal
p <= 0.01. Also reported: how the generator's own combined p-values are distributed on clean windows (before any
conformal correction), i.e. how far the generator alone can be trusted.

Usage: python -m experiments.pilot_power --dataset BN_BTCUSDT --n-a 600 --n-b 600 --n-e 600 --m 200
"""

import argparse
import json
import os
import time

import numpy as np

from lobsurv import injection as inj
from lobsurv.null_model import GeneratorNull, generator_provenance
from lobsurv.scoring import evaluate_mixture, generator_pvalues, score, stat_matrix, uncond_pvalues
from lobsurv.statistics import LOG_DT, make_panel
from lobsurv.fdr import conformal_pvalues

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = ("spoof", "stuffing", "ignition")
SETTINGS = {"spoof": {"k": 10, "tau": 20}, "stuffing": {"n": 25}, "ignition": {"m": 3, "r": 10}}   # mid strength
Q, PREVALENCE = 0.10, 0.05


def fast_threshold(log_dt, q=0.05, floor=np.log(1e-6), resolution=1e-3):
    """
    log inter-event time at or below which an event counts as fast (statistic n_fast). The 5% quantile, unless
    that is the floor value of same-timestamp events (millisecond timestamps, as on Binance): the generator draws
    continuous times and almost never hits the floor exactly, so there the threshold is half the timestamp
    resolution, i.e. "in the same millisecond", which treats real and generated times alike.
    """
    t = float(np.quantile(log_dt, q))
    return float(np.log(resolution / 2)) if t <= floor + 1e-9 else t


def day_positions(null, horizon, context):
    """Non-overlapping window starts per validation day (stream positions)."""
    v0, n = null.offsets["val"], len(null.stream)
    starts = list(null.G["val_segments"]) if null.G["val_segments"] is not None else [0]
    bounds = [v0 + s for s in starts] + [n]
    return [np.arange(a + context, b - horizon, horizon) for a, b in zip(bounds[:-1], bounds[1:])]


def inject(family, P, prev, rng, units):
    t0 = int(rng.integers(10, 61))
    if family == "spoof":
        return inj.spoof(P, prev, t0, "bid" if rng.random() < 0.5 else "ask", units=units, **SETTINGS[family])[0]
    if family == "stuffing":
        return inj.stuffing(P, prev, t0, units=units, rng=rng, **SETTINGS[family])[0]
    return inj.ignition(P, prev, t0, direction=1 if rng.random() < 0.5 else -1, units=units, **SETTINGS[family])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BN_BTCUSDT")
    ap.add_argument("--n-a", type=int, default=600)
    ap.add_argument("--n-b", type=int, default=600)
    ap.add_argument("--n-e", type=int, default=600)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    null = GeneratorNull(a.dataset, parts=("train", "val"))
    S, view, H = null.stream, null.view, null.horizon
    train_phys = view.to_phys(S[:null.G["n_train"]])
    fast = fast_threshold(train_phys[:, LOG_DT])
    units = inj.Units.from_train(train_phys)
    days = day_positions(null, H, null.context)
    assert len(days) >= 4, f"need at least 4 validation days, got {len(days)}"
    pick = lambda pos, k: np.sort(rng.choice(pos, size=min(k, len(pos)), replace=False))
    A, B, E = pick(np.concatenate(days[:2]), a.n_a), pick(days[2], a.n_b), pick(np.concatenate(days[3:]), a.n_e)

    def panel_at(s):
        return make_panel(view, null.has_dt, S[s - 1], fast_log_dt=fast)

    names = list(panel_at(A[0]))
    ref = np.concatenate([stat_matrix(panel_at(s), S[s:s + H][None], names) for s in A])

    def variants(s):
        clean = S[s:s + H]
        P, prev = view.to_phys(clean), view.to_phys(S[s - 1])
        return np.stack([clean] + [view.to_norm(inject(f, P, prev, rng, units)) for f in FAMILIES])

    out = {}
    for label, positions, make in (("B", B, lambda s: S[s:s + H][None]), ("E", E, variants)):
        gp, up = [], []
        t0 = time.perf_counter()
        for i, s in enumerate(positions):
            panel = panel_at(s)
            obs = stat_matrix(panel, make(s), names)
            sim = stat_matrix(panel, null.simulate(s, a.m, rng), names)
            gp.append(generator_pvalues(sim, obs, rng=rng))
            up.append(uncond_pvalues(ref, obs, rng=rng))
            if i % 100 == 0:
                print(f"[{a.dataset}] {label} {i}/{len(positions)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        out[label] = {"gen": np.array(gp), "unc": np.array(up)}       # [n, variants, stats]

    res = {"dataset": a.dataset, "seed": a.seed, "m": a.m, "q": Q, "prevalence": PREVALENCE, "statistics": names,
           "settings": SETTINGS, "n": {"A": len(A), "B": len(B), "E": len(E)}, "provenance": generator_provenance(),
           "detectors": {}}
    for det in ("gen", "unc"):
        cal = score(out["B"][det][:, 0])
        sc = score(out["E"][det])                                      # [n_E, 1 + families]
        r = {}
        for fi, fam in enumerate(FAMILIES, start=1):
            mix = evaluate_mixture(cal, sc[:, 0], sc[:, fi], Q, PREVALENCE, np.random.default_rng(a.seed + fi), reps=50)
            p_conf = conformal_pvalues(cal, sc[:, fi])
            mix["share_p_le_0.01"] = float(np.mean(p_conf <= 0.01))
            mix["median_stat_p"] = dict(zip(names, np.median(out["E"][det][:, fi], axis=0).round(4).tolist()))
            r[fam] = mix
        r["clean_false_alarm_p_le_0.01"] = float(np.mean(conformal_pvalues(cal, sc[:, 0]) <= 0.01))
        res["detectors"][det] = r
    comb = np.exp(-score(out["E"]["gen"][:, 0]))                        # generator's own combined p, clean windows
    res["generator_raw_calibration"] = {f"share_p_le_{t}": float(np.mean(comb <= t)) for t in (0.01, 0.05, 0.1, 0.5)}
    res["seconds"] = time.perf_counter() - t_start
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"pilot_power_{a.dataset}_seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    for det in ("gen", "unc"):
        print(det, {f: (round(v["power"], 3), round(v["fdr"], 3), round(v["share_p_le_0.01"], 3))
                    for f, v in res["detectors"][det].items() if isinstance(v, dict)})
    print("generator raw calibration on clean windows:", res["generator_raw_calibration"])


if __name__ == "__main__":
    main()
