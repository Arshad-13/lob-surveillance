"""
Every number and table of the paper (manuscript/), generated from the result files only.

  manuscript/generated/numbers.tex   number macros, one newcommand per value (macro names are letters only)
  manuscript/generated/table_*.tex   tables

Usage: python -m experiments.paper_assets   (after experiments.followup q1 [and q2])
"""

import json
import os

import numpy as np

from experiments.followup import FAMILIES, RES, SYMBOLS, load, short

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "manuscript", "generated")
FAM_TEX = {"spoof": "Spoofing", "stuffing": "Quote stuffing", "ignition": "Momentum ignition"}
WORD = {"spoof": "Spoof", "stuffing": "Stuff", "ignition": "Ignit"}
SYM_WORD = {s: short(s).capitalize() for s in SYMBOLS}


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.0f}"


def write(name, body):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(body.rstrip() + "\n")


def stage0(macros):
    runs = {ds: load(ds) for ds in SYMBOLS}
    d0 = runs[SYMBOLS[0]]
    macros.update({"nSymbols": len(SYMBOLS), "nRef": d0["n"]["A"], "nCal": d0["n"]["B"], "nEval": d0["n"]["E"],
                   "Msim": d0["m"], "Mrefine": d0["m_refine"], "nStats": len(d0["statistics"]),
                   "nVariants": len(d0["variants"]) - 1, "seedUsed": d0["seed"],
                   "nGridSpoof": len(d0["grid"]["spoof"]), "nGridStuff": len(d0["grid"]["stuffing"]),
                   "nGridIgnit": len(d0["grid"]["ignition"])})
    refB = [runs[ds]["n_refined"]["B"] / runs[ds]["n"]["B"] for ds in SYMBOLS]
    refE = [runs[ds]["n_refined"]["E"] / runs[ds]["n"]["E"] for ds in SYMBOLS]
    macros.update({"refinedCalLo": pct(min(refB)), "refinedCalHi": pct(max(refB)),
                   "refinedEvalLo": pct(min(refE)), "refinedEvalHi": pct(max(refE))})
    hours = [runs[ds]["seconds"] / 3600 for ds in SYMBOLS]
    macros.update({"hoursLo": f"{min(hours):.1f}", "hoursHi": f"{max(hours):.1f}"})

    def fam_power(ds, det, al):
        pw = runs[ds]["detectors"][det]["power"]
        return {f: float(np.mean([v[str(al)] for k, v in pw.items() if k.startswith(f + ":")])) for f in FAMILIES}

    pooled = {det: {f: float(np.mean([fam_power(ds, det, 0.05)[f] for ds in SYMBOLS])) for f in FAMILIES}
              for det in ("gen", "unc")}
    for f in FAMILIES:
        macros[f"powGen{WORD[f]}"] = f3(pooled["gen"][f])
        macros[f"powUnc{WORD[f]}"] = f3(pooled["unc"][f])
    fa = {det: {ds: runs[ds]["detectors"][det]["false_alarm"]["0.05"] for ds in SYMBOLS} for det in ("gen", "unc")}
    for ds in SYMBOLS:
        macros[f"faGen{SYM_WORD[ds]}"] = f3(fa["gen"][ds])
        macros[f"faUnc{SYM_WORD[ds]}"] = f3(fa["unc"][ds])
    macros["faUncAtMostNominal"] = sum(v <= 0.05 for v in fa["unc"].values())
    with open(os.path.join(RES, f"pilot_power_{SYMBOLS[0]}_seed0.json")) as f:
        macros["nCalFirstPilot"] = json.load(f)["n"]["B"]
    two_se = 0.05 + 2 * np.sqrt(0.05 * 0.95 / d0["n"]["E"])
    macros["faTwoSE"] = f3(two_se)
    macros["faGenAboveTwoSE"] = sum(v > two_se for v in fa["gen"].values())
    macros["faUncAboveTwoSE"] = sum(v > two_se for v in fa["unc"].values())
    macros["gapSpoof"] = f2(pooled["gen"]["spoof"] - pooled["unc"]["spoof"])

    # Table: Stage-0 power at alpha = 0.05 per family and symbol, and the clean false-alarm rate
    cols = " & ".join(short(ds) for ds in SYMBOLS)
    lines = [r"\begin{tabular}{l" + "c" * (len(SYMBOLS) + 1) + "}", r"\toprule",
             f"& {cols} & Mean" + r" \\", r"\midrule"]
    for f in FAMILIES:
        for det, lab in (("gen", "cond."), ("unc", "uncond.")):
            vals = [fam_power(ds, det, 0.05)[f] for ds in SYMBOLS]
            name = f"{FAM_TEX[f]}, {lab}" if det == "gen" else f"\\quad {lab}"
            lines.append(name + " & " + " & ".join(f2(v) for v in vals) + f" & {f3(pooled[det][f])}" + r" \\")
    lines.append(r"\midrule")
    for det, lab in (("gen", "False alarm, cond."), ("unc", r"\quad uncond.")):
        lines.append(lab + " & " + " & ".join(f3(fa[det][ds]) for ds in SYMBOLS) +
                     f" & {f3(np.mean(list(fa[det].values())))}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_stage0", "\n".join(lines))


def q1(macros):
    p = os.path.join(RES, "followup_q1.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        q = json.load(f)
    for f in FAMILIES:
        for det, D in (("gen", "Gen"), ("unc", "Unc")):
            macros[f"mpow{D}{WORD[f]}"] = f3(q["pooled"]["matched_power_0.05"][det][f])
            macros[f"mpowStrict{D}{WORD[f]}"] = f3(q["pooled"]["matched_power_0.01"][det][f])
            macros[f"auc{D}{WORD[f]}"] = f3(q["pooled"]["auc"][det][f])
        macros[f"mpowSymWins{WORD[f]}"] = q["family_wins"][f]["symbols_at_least_as_high"]
    macros["qOneHolds"] = "holds" if q["q1_holds"] else "fails"
    mp = q["pooled"]["matched_power_0.05"]
    macros["mgapSpoof"] = f2(mp["gen"]["spoof"] - mp["unc"]["spoof"])

    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Matched, $\alpha=0.05$} & \multicolumn{2}{c}{Matched, $\alpha=0.01$} & "
             r"\multicolumn{2}{c}{AUC} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Family & cond. & uncond. & cond. & uncond. & cond. & uncond. \\", r"\midrule"]
    for f in FAMILIES:
        cells = []
        for key in ("matched_power_0.05", "matched_power_0.01", "auc"):
            g, u = q["pooled"][key]["gen"][f], q["pooled"][key]["unc"][f]
            cells += [(r"\textbf{" + f3(g) + "}") if g > u else f3(g), (r"\textbf{" + f3(u) + "}") if u > g else f3(u)]
        lines.append(FAM_TEX[f] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_matched", "\n".join(lines))


ARM_WORD = {"original": "Orig", "V1": "VOne", "V2": "VTwo", "V3": "VThree"}
ARM_TEX = {"original": "Original", "V1": "V1 matched resolution", "V2": "V2 rolling calibration", "V3": "V3 both"}


def q2(macros):
    p = os.path.join(RES, "followup_q2.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        q = json.load(f)
    assert q["integrity_ok"]
    ps = q["per_symbol"]
    macros["integrityMaxDiff"] = f"{max(q['integrity'].values()):.0e}" if max(q["integrity"].values()) else "0"
    refV1 = [ps[ds]["refined_share"]["E_clean_V1"] for ds in SYMBOLS]
    macros.update({"refinedEvalVOneLo": pct(min(refV1)), "refinedEvalVOneHi": pct(max(refV1))})
    n_cal = [n for ds in SYMBOLS for d, n in ps[ds]["n_cal_rolling"].items() if d != min(ps[ds]["n_cal_rolling"])]
    macros.update({"nCalRollLo": min(n_cal), "nCalRollHi": max(n_cal)})
    two_se = 0.05 + 2 * np.sqrt(0.05 * 0.95 / ps[SYMBOLS[0]]["V3"]["gen"]["n"])

    def fa(arm, det, ds):
        return ps[ds][arm][det]["false_alarm"]["0.05"]

    def pw(arm, det, key):
        return {f: float(np.mean([ps[ds][arm][det][key][f] for ds in SYMBOLS])) for f in FAMILIES}

    rows = []
    for arm, A in ARM_WORD.items():
        cells = []
        for det, D in (("gen", "Gen"), ("unc", "Unc")):
            vals = [fa(arm, det, ds) for ds in SYMBOLS]
            macros[f"faMax{D}{A}"] = f3(max(vals))
            macros[f"faMean{D}{A}"] = f3(np.mean(vals))
            macros[f"nAbove{D}{A}"] = sum(v > two_se for v in vals)
            for ds in SYMBOLS:
                macros[f"fa{A}{D}{SYM_WORD[ds]}"] = f3(fa(arm, det, ds))
        p5 = {det: pw(arm, det, "power_0.05") for det in ("gen", "unc")}
        for f in FAMILIES:
            g, u = p5["gen"][f], p5["unc"][f]
            macros[f"pow{A}Gen{WORD[f]}"], macros[f"pow{A}Unc{WORD[f]}"] = f3(g), f3(u)
            cells += [(r"\textbf{" + f3(g) + "}") if g > u else f3(g), (r"\textbf{" + f3(u) + "}") if u > g else f3(u)]
        fa_cells = [f3(max(fa(arm, det, ds) for ds in SYMBOLS)) for det in ("gen", "unc")]
        fa_cells = [(r"\underline{" + c + "}") if float(c) > 0.10 else c for c in fa_cells]
        rows.append(ARM_TEX[arm] + " & " + " & ".join(fa_cells) + " & " + " & ".join(cells) + r" \\")
    p1 = {det: pw("V3", det, "power_0.01") for det in ("gen", "unc")}
    for f in FAMILIES:
        macros[f"powStrictVThreeGen{WORD[f]}"] = f3(p1["gen"][f])
        macros[f"powStrictVThreeUnc{WORD[f]}"] = f3(p1["unc"][f])
    macros["rgapSpoof"] = f2(pw("V3", "gen", "power_0.05")["spoof"] - pw("V3", "unc", "power_0.05")["spoof"])
    macros["repairedDesign"] = q["repaired_design"] or "none"
    macros["repairedWins"] = q.get("repaired_power_wins", 0)
    macros["qTwoOutcome"] = q["outcome"]
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Max.\ false alarm} & \multicolumn{2}{c}{Spoofing} & \multicolumn{2}{c}{Stuffing} & "
             r"\multicolumn{2}{c}{Ignition} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Arm & cond. & uncond. & cond. & uncond. & cond. & uncond. & cond. & uncond. \\", r"\midrule"]
    lines += rows + [r"\bottomrule", r"\end{tabular}"]
    write("table_q2", "\n".join(lines))

    # Table: the conditional detector's false-alarm rate by evaluation day, original rule and V3
    days = sorted(int(d) for d in ps[SYMBOLS[0]]["false_alarm_by_day"]["original"]["gen"])
    head = " & ".join(str(d + 1) for d in days)
    lines = [r"\begin{tabular}{l" + "c" * (2 * len(days)) + "}", r"\toprule",
             rf"& \multicolumn{{{len(days)}}}{{c}}{{Original}} & \multicolumn{{{len(days)}}}{{c}}{{V3}} \\",
             rf"\cmidrule(lr){{2-{1 + len(days)}}}\cmidrule(lr){{{2 + len(days)}-{1 + 2 * len(days)}}}",
             f"Day & {head} & {head}" + r" \\", r"\midrule"]
    for ds in SYMBOLS:
        by = ps[ds]["false_alarm_by_day"]
        lines.append(short(ds) + " & " + " & ".join(f3(by[arm]["gen"][str(d)]) for arm in ("original", "V3")
                                                     for d in days) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_byday", "\n".join(lines))
    for arm, A in (("original", "Orig"), ("V3", "VThree")):
        for i, d in enumerate(days):
            macros[f"faDay{A}Xrp" + "ABC"[i]] = f3(ps["BN_XRPUSDT"]["false_alarm_by_day"][arm]["gen"][str(d)])


def confirm(macros, path=None):
    """Test-day confirmation (results/test_confirmation.md): macros, power and false-alarm table, false alarms by day."""
    path = path or os.path.join(RES, "confirm_test_decision.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        c = json.load(f)
    syms, ps = c["symbols"], c["per_symbol"]
    macros.update({"confOutcome": c["outcome"], "confIntegrity": "match" if c["integrity_ok"] else "differ",
                   "confCOne": "holds" if c["c1"] else "fails", "confCTwo": "holds" if c["c2"] else "fails",
                   "confNConfirmed": len(c["confirmed_families"])})
    n = [ps[ds]["gen"]["n"] for ds in syms]                         # clean evaluation windows per symbol
    hours = [ps[ds]["hours"] for ds in syms]
    macros.update({"nTestWinLo": min(n), "nTestWinHi": max(n),
                   "confHoursLo": f"{min(hours):.1f}", "confHoursHi": f"{max(hours):.1f}"})
    for key, K in (("calibration", "Cal"), ("eval_clean", "Clean"), ("eval_injected", "Inj")):
        v = [ps[ds]["refined_share"][key] for ds in syms]
        macros[f"refinedTest{K}Lo"], macros[f"refinedTest{K}Hi"] = pct(min(v)), pct(max(v))

    def fa(det, ds, al="0.05"):
        return ps[ds][det]["false_alarm"][al]

    for det, D in (("gen", "Gen"), ("unc", "Unc")):
        for ds in syms:
            macros[f"faConf{D}{SYM_WORD[ds]}"] = f3(fa(det, ds))
        vals = [fa(det, ds) for ds in syms]
        macros[f"faConfMax{D}"], macros[f"faConfMean{D}"] = f3(max(vals)), f3(np.mean(vals))
        macros[f"faConfStrictMax{D}"] = f3(max(fa(det, ds, "0.01") for ds in syms))
        above = [short(ds) for ds in syms if fa(det, ds) > 0.05 + 2 * np.sqrt(0.05 * 0.95 / ps[ds][det]["n"])]
        macros[f"nAboveTwoSEConf{D}"] = len(above)
        macros[f"aboveTwoSEConf{D}"] = ", ".join(above) if above else "none"
        macros[f"faConfDayMax{D}"] = f3(max(v for ds in syms for v in ps[ds][det]["false_alarm_by_day"].values()))
        for f in FAMILIES:
            for key, K in (("power_0.05", "pow"), ("power_0.01", "powStrict"), ("matched_power_0.05", "mpow"),
                           ("matched_power_0.01", "mpowStrict"), ("auc", "auc"), ("bh", "bh"), ("bh_fdr", "fdr")):
                macros[f"{K}Conf{D}{WORD[f]}"] = f3(c["pooled"][key][det][f])
    for f in FAMILIES:
        for al, A in (("0.05", ""), ("0.01", "Strict")):
            d = c["power_difference"][al][f]
            macros[f"dConf{A}{WORD[f]}"] = f3(d["estimate"])
            macros[f"dConf{A}Lo{WORD[f]}"], macros[f"dConf{A}Hi{WORD[f]}"] = f3(d["lo"]), f3(d["hi"])
            macros[f"dConf{A}Syms{WORD[f]}"] = d["symbols_gen_higher"]

    # Table: power at alpha = 0.05 per family and symbol, the pooled difference with its interval, false alarms
    cols = " & ".join(short(ds) for ds in syms)
    lines = [r"\begin{tabular}{l" + "c" * (len(syms) + 1) + "}", r"\toprule", f"& {cols} & Mean" + r" \\", r"\midrule"]
    for f in FAMILIES:
        for det, lab in (("gen", "cond."), ("unc", "uncond.")):
            vals = [ps[ds][det]["power_0.05"][f] for ds in syms]
            name = f"{FAM_TEX[f]}, {lab}" if det == "gen" else f"\\quad {lab}"
            lines.append(name + " & " + " & ".join(f2(v) for v in vals) + f" & {f3(np.mean(vals))}" + r" \\")
        d = c["power_difference"]["0.05"][f]
        lines.append(rf"\quad difference & \multicolumn{{{len(syms)}}}{{r}}{{95\% interval {f3(d['lo'])} to "
                     rf"{f3(d['hi'])}}} & {f3(d['estimate'])}" + r" \\")
    lines.append(r"\midrule")
    for det, lab in (("gen", "False alarm, cond."), ("unc", r"\quad uncond.")):
        vals = [fa(det, ds) for ds in syms]
        lines.append(lab + " & " + " & ".join(f3(v) for v in vals) + f" & {f3(np.mean(vals))}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_confirm", "\n".join(lines).replace("-0.", "$-$0."))

    # Table: false-alarm rate at alpha = 0.05 by test day, both detectors
    days = sorted(int(d) for d in ps[syms[0]]["gen"]["false_alarm_by_day"])
    lines = [r"\begin{tabular}{l" + "c" * len(days) + "}", r"\toprule",
             "Test day & " + " & ".join(f"T{d}" for d in days) + r" \\", r"\midrule"]
    for det, lab in (("gen", "Conditional"), ("unc", "Unconditional")):
        lines.append(rf"\multicolumn{{{1 + len(days)}}}{{l}}{{\emph{{{lab}}}}} \\")
        for ds in syms:
            by = ps[ds][det]["false_alarm_by_day"]
            lines.append(short(ds) + " & " + " & ".join(f3(by[str(d)]) for d in days) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_confbyday", "\n".join(lines))


def main():
    macros = {}
    stage0(macros)
    q1(macros)
    q2(macros)
    confirm(macros)
    body = ["% generated by experiments/paper_assets.py from results/; do not edit"]
    for k, v in macros.items():
        assert k.isalpha(), k
        body.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    write("numbers", "\n".join(body))
    print(f"wrote {len(macros)} macros to {GEN}")


if __name__ == "__main__":
    main()
