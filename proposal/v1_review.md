# Review of the v1 proposal (SMFI, August 2026) and what v2 changes

v1 is kept unchanged in `archive/v1_smfi/` (PDF, long markdown version, engine code).
Review date: 2026-10-21. The v2 proposal is `proposal/proposal.md`.

## Problems that made v1 unrunnable or its results meaningless

| # | v1 | Why it fails | v2 fix |
|---|---|---|---|
| 1 | Benjamini-Hochberg over the 200 injected events, "FDR" = false-discovery proportion of that list | Every tested event is a real injection, so no discovery can be false: the headline FDR is 0 by construction | Test **every** window (clean and injected, ~5% prevalence); FDR and power measured against known labels; BH on non-overlapping windows, BY where windows overlap |
| 2 | p_i = #{clean ≥ obs} / N_clean | Can be exactly 0 (not a valid p-value); BH's first threshold 0.05/200 needs N_clean ≥ 4,000 | Conformal p-values (1 + #) / (n + 1) (Bates et al. 2023), which BH provably controls; generator p-values use the same +1 form |
| 3 | Unconditional null: "clean windows" pooled over the day | Busy periods trip the alarm, manipulation in quiet periods hides; this confound is what the generator work measured (intraday regime drift) | **Conditional null**: the frozen companion generator simulates each window's continuation from its own context |
| 4 | "Zigzag persistence" + SMFI(t) = W2(D_t, D_{t−1}); union-only zigzag (PDF eq. 2) vs union and intersection (markdown) | Zigzag gives one barcode for the whole sequence, not per-window diagrams. The engine computed ordinary per-window Rips persistence, no zigzag | Say what it is: sliding-window Rips persistence. It is one candidate statistic, not the method (H4) |
| 5 | β0 signal as the primary topological feature | H0 of a Rips filtration = minimum-spanning-tree edge lengths = single linkage; it is a nearest-neighbour statistic, not topology | H0 is treated as the kNN/MST control; only H1 can add anything topological, and H4 tests whether it does |
| 6 | FI-2010 merged into Tier 1 "to guarantee N_inj ≥ 200" | Public FI-2010 has no timestamps and no message stream: no 35 s windows, no cancels, no VPIN. The generator also never passes MMD on it | FI-2010 dropped. It was not needed: one LOBSTER day has ~4,000 non-overlapping 100-event windows per ticker, and the Binance panel has 30 days × 6 symbols |
| 7 | Tier 2: 2010 Flash Crash, Coscia, Do & Putniņš cases at $0 | Flash Crash and Coscia are CME futures (not in LOBSTER's NASDAQ sample); prosecuted-case data are proprietary | Dropped as a tier. Real data are used for negative controls and an audit of top discoveries, with no manipulation claim |
| 8 | Tier 3 CAEP: min over generator θ of E[SMFI(Gθ(z))] s.t. M ≥ τ, with CT-ESN as G | The generator is level-1 (mid move, spread, best-queue sizes, inter-event time): it cannot place layered orders. SMFI has no usable gradient. τ came from replay injections, which have no market response, so the price-impact part of M ≈ 0 | Replaced by an **evasion frontier** (M1): a grid over manipulation parameters, measuring detection power and economic effect; the effect needs the generator's reactive response, and is reported only if the response passes a realism check (R) |
| 9 | Multiparameter persistence for k = 3 tickers, "EXPSPACE" | 3 vertices give ≤ 3 edges and 1 triangle: equivalent to thresholding three distance series; the complexity framing over-claims (the rank invariant is cheap) | Cross-asset becomes a p-value combination across symbols (Cauchy combination, valid under dependence) on the Binance panel, with injections too weak to detect on any one symbol (H5, exploratory) |
| 10 | Cross-paper link through coupled-reservoir β_ij | The coupled multi-asset hypothesis of the generator paper was not supported | Link removed; the generator is used single-asset only |
| 11 | Log-signature depth 4 on mid, imbalance and top-N volumes | ~5,800 dimensions for 12 channels; Rips on 5–50 points there suffers distance concentration; normalisation unspecified; N used for four things | 4 channels at depth 3 (30 dimensions), z-scored with training statistics |
| 12 | Lead time with a −5 s early credit | Injections have no "collapse" to lead; the early credit counts noise alarms as detections; window length in seconds was undefined | Detection delay from injection start to first discovery, in events and seconds; no early credit |
| 13 | Hypothesis says "structural liquidity collapse", tests spoofing; name "Systemic Market Fragility Index" | Two different targets; "systemic" was only ever 3 tickers | One target: unusual order flow, with manipulation footprints as the labelled test cases |
| 14 | 140 GPU-hours | The persistence and the generator's neighbour search are CPU code | Budget in CPU-hours, measured in Stage 0; GPU only if the neighbour search is ported (planned in the generator repo) |
| 15 | Unsourced specifics: injection moments "match the first two moments of Do & Putniņš", "30 s enforcement-standard window" | Neither could be traced to a source | Removed. Injection sizes are set in the data's own units (multiples of median queue size, event rate) |
| 16 | Engine: blanket `except` falls back to a different metric (centroid distance, hand-rolled H0) | A missing library silently changes the reported number | `lobsurv/persistence.py` raises instead; the archived engine is not used |

## Reference corrections

Checked by web search on 2026-10-21.

| v1 citation | Finding | Action |
|---|---|---|
| Briola et al., arXiv:2401.08274 | Wrong ID: HLOB is arXiv:2405.18938, Expert Syst. Appl. 266 (2025) 126078 | Fixed |
| Turiel et al., *Quantitative Finance* 2022 | It is *Entropy* 2022; it measures persistence of TMFG motifs, not persistent homology | Fixed and re-described |
| Fabre & Challet, arXiv:2501.04912 | ID not found; the paper is arXiv:2504.15908 (neural network over Hawkes features) | Fixed |
| Lin & Yang, *Expert Syst. Appl.* 2025 | Authors Yushi Lin & Peng Yang, arXiv:2508.17086; journal not confirmed | Cited as arXiv |
| Cartea et al., *Math. Finance* 2020 | It is *Applied Mathematical Finance* 27(1–2):67–98 | Fixed |
| Tao et al., *Quant. Finance* 2020 | Quant. Finance 22(8):1405–1425, **2022** (arXiv 2020) | Fixed |
| Cartea, Chang & García-Arenas 2023, SSRN:4432128 | Paper exists (Oxford working paper); SSRN ID not confirmed | Cited as working paper; verify ID |
| Kularatnam & Stathaki 2024 | arXiv:2403.13429, AAAI 2024 workshop | Fixed |
| **Zheng (2025), IEEE TKDE** | **Not found** | Removed |
| **Do & Putniņš (2023), J. Financ. Econ.** | **Not found** | Removed |
| Li, Polukarova & Ventre 2023 | Probably arXiv:2308.08683; authors not confirmed | Not cited until confirmed |
