"""
Stage-0 pilot, redesigned (results/stage0_redesign.md): detection at fixed false-alarm rates across a grid of
injection strengths, generator-conditional against unconditional null. Validation data only.

Validation days of one symbol, by day:
  A  every window of days 1-2: reference set of the unconditional null
  B  every window of day 3: clean calibration windows for the conformal step, shared by both detectors
  E  every window of days 4-6: scored clean and with every injection of the grid (same position, side and
     direction for all strengths of a family in one window, so strengths are paired)
Windows are non-overlapping 100-event stretches, each with the 100 real events before it as context.

Reported per detector, family and strength: the share of injected windows with conformal p <= alpha for
alpha in ALPHAS (power at a fixed false-alarm rate), next to the clean windows' share (the realised
false-alarm rate). Secondary: power and false-discovery proportion of BH at q = 0.10 on mixtures with 5%
prevalence, at every strength. Per-window conformal p-values are saved, so every number can be recomputed.

Usage: python -m experiments.pilot_strength --dataset BN_BTCUSDT [--max-windows 300 --m 50 for a smoke test]
"""

import argparse
import json
import os
import time

import numpy as np

from experiments.pilot_power import day_positions, fast_threshold
from lobsurv import injection as inj
from lobsurv.fdr import conformal_pvalues
from lobsurv.null_model import GeneratorNull, generator_provenance
from lobsurv.scoring import evaluate_mixture, generator_pvalues, score, stat_matrix, uncond_pvalues
from lobsurv.statistics import LOG_DT, make_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = {"spoof": [{"k": k, "tau": 20} for k in (5, 10, 20, 50, 100)],
        "stuffing": [{"n": n} for n in (10, 25, 50, 100)],
        "ignition": [{"m": m, "r": 10} for m in (1, 2, 3, 5)]}
ALPHAS, Q, PREVALENCE = (0.01, 0.05), 0.10, 0.05


def label(family, s):
    return family + ":" + ",".join(f"{k}={v}" for k, v in s.items())


VARIANTS = ["clean"] + [label(f, s) for f, grid in GRID.items() for s in grid]


def variants(P, prev, rng, units):
    out = [P]
    for fam, grid in GRID.items():
        t0 = int(rng.integers(10, 61))
        side, direction = ("bid" if rng.random() < 0.5 else "ask"), (1 if rng.random() < 0.5 else -1)
        for s in grid:
            if fam == "spoof":
                out.append(inj.spoof(P, prev, t0, side, units=units, **s)[0])
            elif fam == "stuffing":
                out.append(inj.stuffing(P, prev, t0, units=units, rng=np.random.default_rng(t0), **s)[0])
            else:
                out.append(inj.ignition(P, prev, t0, direction=direction, units=units, **s)[0])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BN_BTCUSDT")
    ap.add_argument("--max-windows", type=int, default=0, help="cap per set (0: every window)")
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--m-refine", type=int, default=1000, help="continuations when a window beats almost all of --m")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.perf_counter()
    provenance = generator_provenance()
    rng = np.random.default_rng(a.seed)
    null = GeneratorNull(a.dataset, parts=("train", "val"))
    S, view, H = null.stream, null.view, null.horizon
    train_phys = view.to_phys(S[:null.G["n_train"]])
    fast, units = fast_threshold(train_phys[:, LOG_DT]), inj.Units.from_train(train_phys)
    days = day_positions(null, H, null.context)
    assert len(days) >= 4, f"need at least 4 validation days, got {len(days)}"
    cap = lambda pos: pos if not a.max_windows or len(pos) <= a.max_windows else np.sort(
        rng.choice(pos, a.max_windows, replace=False))
    A, B, E = cap(np.concatenate(days[:2])), cap(days[2]), cap(np.concatenate(days[3:]))
    panel_at = lambda s: make_panel(view, null.has_dt, S[s - 1], fast_log_dt=fast)
    names = list(panel_at(A[0]))
    ref = np.concatenate([stat_matrix(panel_at(s), S[s:s + H][None], names) for s in A])

    pv, n_refined = {}, {"B": 0, "E": 0}
    for set_name, positions in (("B", B), ("E", E)):
        gp, up = [], []
        t0 = time.perf_counter()
        for i, s in enumerate(positions):
            panel = panel_at(s)
            if set_name == "B":
                windows = S[s:s + H][None]
            else:
                windows = view.to_norm(variants(view.to_phys(S[s:s + H]), view.to_phys(S[s - 1]), rng, units))
            obs = stat_matrix(panel, windows, names)
            sim = stat_matrix(panel, null.simulate(s, a.m, rng), names)
            g = generator_pvalues(sim, obs, rng=rng)
            if a.m_refine > a.m and (g <= 2.0 / (a.m + 1)).any():
                # some variant beats (nearly) every continuation: simulate more, so the generator's p-values reach a
                # resolution comparable to the unconditional null's reference set (same rule for B and E)
                sim = np.concatenate([sim, stat_matrix(panel, null.simulate(s, a.m_refine - a.m, rng), names)])
                g = generator_pvalues(sim, obs, rng=rng)
                n_refined[set_name] += 1
            gp.append(g)
            up.append(uncond_pvalues(ref, obs, rng=rng))
            if i % 200 == 0:
                print(f"[{a.dataset}] {set_name} {i}/{len(positions)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        pv[set_name] = {"gen": np.array(gp), "unc": np.array(up)}       # [n, variants, statistics]

    res = {"dataset": a.dataset, "seed": a.seed, "m": a.m, "m_refine": a.m_refine, "n_refined": n_refined,
           "statistics": names, "grid": GRID, "variants": VARIANTS,
           "alphas": ALPHAS, "q": Q, "prevalence": PREVALENCE, "n": {"A": len(A), "B": len(B), "E": len(E)},
           "fast_log_dt": fast, "provenance": provenance, "detectors": {}}
    for det in ("gen", "unc"):
        cal = score(pv["B"][det][:, 0])
        sc = score(pv["E"][det])                                             # [n_E, variants]
        p = np.stack([conformal_pvalues(cal, sc[:, v]) for v in range(len(VARIANTS))], axis=1)
        r = {"false_alarm": {str(al): float(np.mean(p[:, 0] <= al)) for al in ALPHAS}, "power": {}, "bh": {},
             "conformal_p": np.round(p, 5).tolist()}
        for v, name in enumerate(VARIANTS[1:], start=1):
            r["power"][name] = {str(al): float(np.mean(p[:, v] <= al)) for al in ALPHAS}
            r["bh"][name] = evaluate_mixture(cal, sc[:, 0], sc[:, v], Q, PREVALENCE,
                                             np.random.default_rng(a.seed + v), reps=50)
        res["detectors"][det] = r
    comb = np.exp(-score(pv["E"]["gen"][:, 0]))
    res["generator_raw_calibration"] = {f"share_p_le_{t}": float(np.mean(comb <= t)) for t in (0.01, 0.05, 0.1, 0.5)}
    res["seconds"] = time.perf_counter() - t_start
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"pilot_strength_{a.dataset}_seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    for det in ("gen", "unc"):
        print(det, "false alarm", res["detectors"][det]["false_alarm"])
        for name, pw in res["detectors"][det]["power"].items():
            print(f"  {det} {name:22s} power@1% {pw['0.01']:.3f}  @5% {pw['0.05']:.3f}  "
                  f"BH power {res['detectors'][det]['bh'][name]['power']:.3f}")


if __name__ == "__main__":
    main()
