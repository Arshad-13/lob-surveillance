# The Systemic Market Fragility Index (SMFI): Detecting Limit Order Book Manipulation via Signature-Based Zigzag Persistent Homology

**Arshad Khatib**
B.Tech in Artificial Intelligence (5th Semester)
Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
August 2026

---

## Abstract

Topological methods have already been shown to capture exploitable structure in financial data — at the level of correlation networks (Turiel, Barucca & Aste, 2022), macro index dynamics (Gidea & Katz, 2018), and, most closely, within the limit order book itself for price forecasting (HLOB; Briola, Bartolucci & Aste, 2024). What remains open is whether a topological construction operating natively at LOB-event granularity can *detect manipulation* — not just structure or forecast price — with the statistical rigor (FDR, lead-time) that a surveillance tool requires, and whether that construction can be extended beyond a single instrument to genuinely systemic, cross-asset manipulation patterns. This paper proposes the **Systemic Market Fragility Index (SMFI)**: zigzag persistent homology applied to log-signature embeddings of LOB data, benchmarked against volume-based (VPIN, OFI) and structural (motif/graph-based) detectors on False Discovery Rate and detection lead-time; extended to a genuinely systemic, multi-asset setting via a bounded multiparameter persistence construction that tracks topological coupling *between* correlated instruments, not just within one; and stress-tested via a **generalized constrained adversarial evaluation protocol** — a reusable framework for testing any market-surveillance detector against an adaptive, economically-anchored adversary — instantiated here against a reservoir-computing generator.

---

## 1. Core Falsifiable Hypotheses

**Primary hypothesis (single-asset detection):**
> Zigzag persistent homology applied to the log-signatures of Limit Order Book (LOB) data detects structural liquidity collapse with a lower False Discovery Rate (FDR) and earlier detection lead-time than Volume-Synchronized Probability of Informed Trading (VPIN), Order Flow Imbalance (OFI), quote-to-trade ratio, and at least one structural (non-volume-based) 2024–2025 baseline, during large-scale parameterized heuristically injected spoofing events, and generalizes qualitatively to real, prosecuted historical manipulation cases.

**Secondary hypothesis (systemic, cross-asset extension):**
> A bounded multiparameter zigzag persistence construction, tracking topological coupling between a small cluster of correlated instruments over time, detects coordinated cross-asset manipulation episodes (price-linked spoofing across correlated tickers) that single-asset SMFI, run independently on each ticker, misses or detects later.

The secondary hypothesis is explicitly scoped as a bounded, case-study-level claim rather than a fully-powered statistical one (Section 6.3 explains why) — it should not be read as claiming the same statistical rigor as the primary hypothesis.

---

## 2. Related Work

### 2.1 Topology applied to financial data — establishing, then narrowing, the gap
- **Turiel, Barucca & Aste (2022)** and **FinTDA (Souto et al., 2024)** apply persistent/simplicial homology to market-wide correlation structure and general financial time series, respectively, establishing that topological fragility signals exist in financial data generally. Neither operates at order-book-event granularity or targets manipulation specifically.
- **HLOB (Briola, Bartolucci & Aste, 2024)** is the closest existing work: homological methods applied directly to LOB data (via Homological Convolutional Networks / TMFG-based graph filtering) on real LOBSTER-sourced NASDAQ tickers. HLOB establishes that homological structure exists and is exploitable *within* the order book — but for mid-price forecasting, using a fundamentally different topological construction (discretized graph filtering, not continuous zigzag persistence over signature-embedded paths) and without a manipulation-detection framing. **SMFI's precise delta**: same data granularity as HLOB, different topological machinery (zigzag persistence over rough-path log-signatures, chosen specifically for its ability to handle non-monotonic appear/disappear liquidity dynamics without HLOB's discretization), applied to a different question (is this structural change manipulative, not merely predictive of price).

### 2.2 Structural (non-volume-based) spoofing/layering detectors — the baseline set this now includes
- **Zheng (2025)**: temporal motif mining + a motif-aware GNN on LOBSTER data for real-time spoofing/layering detection — the closest *competing* structural framework to SMFI (graphs, not topology, over the same data and problem).
- **Lin & Yang (2025)**: cascaded contrastive representation learning targeting multilevel manipulation specifically, directly analogous to SMFI's parameterized layering-depth injection.
- **Fabre & Challet (2025)**: a Hawkes-process spoofability model, real-time-deployable and interpretable, tested on Level-3 data.
- **Kularatnam & Stathaki (2024)**: weakly-supervised TCN-based spoofing triage.

These four establish that "detect spoofing from LOB structure, not volume aggregates" is an active 2024–2025 research area with several non-topological competitors. **Action taken in this proposal**: at least one of these (Zheng's motif-GNN, or Fabre & Challet's Hawkes model, whichever has usable reference code at the time of the empirical run) is added as a fourth baseline alongside VPIN/OFI/quote-to-trade, so SMFI's central claim is tested against a genuinely current, structurally comparable competitor, not only against an earlier generation of volume-based methods.

### 2.3 Validation against real, prosecuted cases — established practice, not a novel protocol
- **Do & Putniņš (2023)** validate a layering/spoofing detector against a comprehensive, global sample of prosecuted cases — the same protocol idea as SMFI's Tier 2, at meaningfully larger scale than this proposal's two-case check (Flash Crash, Coscia). **This proposal no longer claims novelty for validating against real cases** — that is established practice. Where feasible, Do & Putniņš's case sample is used to expand Tier 2 beyond two events, directly addressing the small-sample limitation flagged in earlier drafts of this proposal.
- **Li, Polukarova & Ventre (2023)** validate a statistical-physics-based structural detector against the 2022 LUNA crash — a second precedent for "structural method + real-crash validation," and a candidate additional Tier 2 event if crypto data (via the Binance extension, Section 5.3) is included in scope.

### 2.4 Adversarial and analytically-modeled spoofing — narrowing Tier 3's novelty claim
- **Cartea, Chang & García-Arenas (2023)** already demonstrate a learning algorithm that adaptively spoofs a real order book under testable emergence conditions. This directly precedes the idea of "an adaptive agent that learns to manipulate an order book" — **SMFI's Tier 3 does not claim novelty for that idea**. What it claims novelty for, more narrowly, is testing that kind of adaptive adversary specifically against a topological detector, under an explicit economic-content constraint that prevents the adversary from trivially "winning" by ceasing to manipulate (Section 4.3) — and, per Section 4.4 below, generalizing that testing procedure into a reusable protocol rather than a one-off check.
- **Cartea, Jaimungal & Wang (2020)** provide an analytical model of how spoofing distorts displayed liquidity and price — used directly to ground the specific functional form of SMFI's manipulation-anchor constraint M(·), rather than that constraint being an arbitrary design choice.
- **Tao, Day, Ling & Drapeau (2020)** model optimal spoofer strategy analytically, providing a second, independent grounding source for M(·)'s threshold calibration.

---

## 3. Problem Formulation & Ground Truth Protocol

Prior literature on LOB manipulation detection largely evaluates via downstream classifiers trained and tested on the same synthetically-labeled data — a circular protocol. This work rejects that in favor of three independent, non-circular tiers, one of which (Tier 2) now draws on a larger external case sample than earlier drafts of this proposal, per Section 2.3.

### 3.1 Tier 1 — Quantitative backbone: parameterized heuristic injection
As in the previous draft: cancel timing drawn from a continuous Poisson process, layering depth varied (1–8 levels), volume asymmetry varied across a realistic range informed by documented enforcement cases (now directly grounded in Cartea, Jaimungal & Wang's and Tao et al.'s analytical spoofing models, Section 2.4, rather than an ad hoc range).

**Design constraint — TSTR generalization**: Tier 1's FDR statistic is computed on synthetic injections, not real labeled manipulation events. This adapts the Train-on-Synthetic, Test-on-Real (TSTR) framework (originally formalized for evaluating the utility of synthetic sequence generators by Esteban et al., 2017) from classifier utility evaluation to detection protocol validation. In our context, this reflects a structural reality: real manipulation episodes (Flash Crash, Coscia, Do & Putniņš' cases) involve mechanisms — correlated cancel cascades, cross-level layering, quote stuffing bursts — that may differ distributionally from the Poisson-timed injection family. This is a *stated design constraint*, not an oversight: genuine real-world manipulation labels are too sparse and heterogeneous to construct a statistically powered FDR test directly. Tier 2 (Section 3.2) serves explicitly as the out-of-sample sanity check for the gap between the injection family and real events.

**Why the injection family is nonetheless a valid proxy**: The injection parameters are not arbitrary — cancel timing (Poisson process with rate calibrated to documented HFT spoofing frequencies) and volume asymmetry range (derived from the analytical distortion function in Cartea, Jaimungal & Wang (2020), which models how a spoofer optimally sizes fictitious orders relative to real execution flow) together define a family whose moments match the first two moments of the empirically documented spoofing parameter distributions in Do & Putniņš (2023). The injection family is not merely "any plausible synthetic manipulation" — it is grounded in the *economics of optimal spoofing*. This grounding is the reason Tier 1 provides meaningful statistical power, not just an arbitrarily benchmarkable label set.

### 3.2 Tier 2 — Qualitative out-of-sample validation, now expanded
The SMFI signal is computed over historical windows surrounding the 2010 Flash Crash (May 6, 2010) and the 2015 Coscia case (August–October 2011), **and, where the underlying case data is accessible, additional events drawn from Do & Putniņš's (2023) documented case sample**, closing the small-sample gap flagged in prior review rounds. As before, this tier does not contribute to the FDR statistic — it is a real-world sanity check.

**Zero data leakage between Tier 1 and Tier 2**: The Tier 1 injected corpus and the Tier 2 historical case set share zero overlapping ticker-dates. Tier 1 injections are applied exclusively to distinct LOBSTER training/evaluation windows outside the dates of these regulatory enforcement actions and market crashes, guaranteeing that the corpus used to calibrate detection thresholds and compute the quantitative FDR statistic is completely disjoint from the historical windows used for qualitative out-of-sample validation.

### 3.3 Tier 3 — Constrained adversarial stress test (instance of the generalized protocol below)
The constrained objective from the previous draft is retained:

```
minimize over generator G:   E_z[ SMFI(G(z)) ]
subject to:                  M(G(z)) >= tau
```

where M(·) — now explicitly grounded in Cartea, Jaimungal & Wang (2020) and Tao et al. (2020) rather than an arbitrary design choice — is a manipulation-anchor score combining cancel-to-execution ratio and realized price impact per unit of displayed-but-unfilled volume, and tau is fixed at the empirical median of these quantities across the Tier 1 injected corpus.

### 3.4 Generalized Constrained Adversarial Evaluation Protocol (CAEP)

Rather than presenting the above as a one-off robustness check specific to SMFI, this proposal formalizes it as a **reusable evaluation protocol applicable to any market-surveillance detector**, positioned as a methodological contribution in its own right, separate from SMFI's detection method itself.

**CAEP, formally**: given any detector D (a function from a market-state path to a fragility/suspicion score), any generator family G_θ capable of producing synthetic market paths, and any economic-content floor M(·) with threshold τ derived from real market data (not from the detector or generator being tested), CAEP evaluates D's adversarial robustness by solving:

```
minimize over θ:   E_z[ D(G_θ(z)) ]
subject to:         M(G_θ(z)) >= tau
```

and reporting (a) the detector's minimum achievable score under the constraint, (b) the constraint-violation rate across candidate generator checkpoints (how often the unconstrained optimum would have been a degenerate non-manipulative solution, had the constraint been absent), and (c) the wall-clock/compute cost of finding a near-optimal adversarial θ, since detectors that are expensive to query (as zigzag persistence is) bound how thoroughly they can be adversarially searched in practice — a property CAEP surfaces explicitly rather than leaving implicit.

**Why this is a genuine, separable contribution**: existing adversarial-spoofing work (Cartea et al., 2023) demonstrates *that* adaptive agents can learn to spoof; it does not propose a general, reusable procedure for evaluating *how robust a given surveillance detector is* to such agents, with an explicit safeguard against the "adversary just stops manipulating" degenerate solution. CAEP is stated generally enough that it could, in future work, be applied to evaluate VPIN, OFI, or Zheng's motif-GNN detector under the same protocol — SMFI is this paper's applied instance of CAEP, not the only thing CAEP could ever test.

**Fallback protocol if CT-ESN fails its primary fidelity hypothesis**: CAEP's reported results are interpretable only if the generator (CT-ESN) produces paths that are realistic enough to constitute a meaningful adversarial challenge — an adversarial search over a low-fidelity generator would be testing the detector's response to noise, not to structured manipulation. If CT-ESN's primary hypothesis (Sig-MMD competitive with Sig-WGAN / TimeGAN, as defined in Section 1 of the companion paper) is not met at the time of the CAEP run, the CAEP Tier 3 is instead instantiated with **Sig-WGAN** (Ni et al., 2020/2022) as the generator Gθ, which has established LOB-fidelity results. This substitution is stated explicitly in the final paper if it occurs, and the CT-ESN result is reported separately as a negative finding rather than omitted. The CAEP protocol (Section 3.4 above) is generator-agnostic by design and remains valid regardless of which generator is used.

---

## 4. Mathematical Methodology (Single-Asset Core)

Unchanged from the previous revision, restated briefly for completeness:

### 4.1 Log-signatures over sparse microsecond intervals
Each interval of LOB state (mid-price, depth-weighted imbalance, top-N-level volumes) is treated as a continuous path and summarized via its truncated log-signature (depth N = 4, chosen via sensitivity ablation across N ∈ {2, 4, 6}).

### 4.2 Zigzag persistent homology over the signature stream

**Simplicial complex construction — fully specified**: Each log-signature vector (Section 4.1, depth N=4, dimensionality d = dim(log-sig) determined by the number of LOB channels and truncation depth) is treated as a point in ℝ^d. Over a sliding window of W consecutive log-signature vectors, a **Vietoris-Rips complex** is constructed at radius ε — two points are connected by an edge if and only if their Euclidean distance in log-signature space falls below ε. The complex is built to the **2-skeleton** (vertices, edges, and triangles; no higher-dimensional simplices), which is sufficient to track β₀ (connected components) and β₁ (independent cycles/loops) while remaining computationally tractable.

**M3 — Window size W — specified, calibrated, and sensitivity-tested**: The sliding window W (number of consecutive log-signature vectors forming each point cloud) is a critical hyperparameter: too small (W ≈ 2–3) and there are too few points for meaningful topological structure; too large (W ≥ 50) and millisecond spoofing bursts are averaged over and lost. W is **grid-searched across {5, 10, 20, 50}** vectors, calibrated simultaneously with ε on the clean calibration split by jointly maximizing the β₀ dynamic range. Sensitivity of FDR and lead-time metrics to the choice of W is reported in the results alongside ε sensitivity, preventing post-hoc qualification of the optimal window.

**Radius ε calibration & sensitivity ablation**: The primary radius ε is chosen via grid search on a held-out, non-injected calibration window by maximizing the median β₀ dynamic range (difference between the 90th and 10th percentile of β₀ across windows), subject to the non-triviality constraint that β₀ > 1 on at least 50% of calibration windows. Calibrating on *clean* data is a deliberate design choice: it ensures the geometric lens is tuned for general topological expressiveness in normal microstructure regimes without leaking synthetic label information into the topological representation itself. However, to guard against the possibility that dynamic range on clean data does not maximally separate injected spoofing from benign flow, we explicitly conduct a **calibration robustness ablation**: a secondary threshold ε* is calibrated via supervised separation on a held-out train/validation split *within* the Tier 1 injected corpus (strictly disjoint from the final test evaluation split), and both FDR and lead-time metrics under ε and ε* are reported side-by-side.

**Zigzag persistence**: Consecutive complexes C(t_k) and C(t_{k+1}) are linked by the inclusion maps of their intersection and union, allowing the zigzag sequence to track topological features that appear and disappear non-monotonically — the setting that standard (Morse) persistence, which requires monotonic nesting, cannot handle. Betti numbers β₀ and β₁ are tracked across the full sequence.

**Which Betti number carries the primary spoofing signal, and why**: The primary signal is expected to reside in **β₀ (connected components)**, for the following structural reason: a spoofing episode inserts a block of orders at unusual depth levels and cancels them within milliseconds, transiently adding points to the log-signature space that are geometrically isolated from the cloud of normal-regime signatures — i.e., they create new connected components that briefly appear and then disappear. β₁ (loops/cycles) tracks more complex topological structure requiring at least three mutually ε-proximate but not mutually co-located points — a pattern more likely to arise from sustained structural change (e.g., a persistent order-book imbalance regime) than from brief cancel-and-repost spoofing cycles. This does not mean β₁ is uninformative — it is tracked and reported in full — but the primary detection hypothesis (FDR and lead-time claims in Section 1) is evaluated on β₀-based SMFI, with β₁-based SMFI reported as a secondary comparison in the results.

### 4.3 Topological distance and the fragility signal
The SMFI signal is the p-th Wasserstein distance (p=2 as the primary setting, p=1 reported as a sensitivity check) between consecutive persistence diagrams (Cohen-Steiner et al., 2007), smoothed by a persistence-lifespan noise filter.

**Lifespan noise filter — precisely defined**: A topological feature (bar in the zigzag barcode) is retained only if its lifespan (death time minus birth time, measured in window units) exceeds a threshold δ. δ is set to the **10th percentile of the empirical lifespan distribution computed on clean non-injected windows** from the calibration set — i.e., the shortest-lived 10% of features observed in normal-market conditions are filtered out as noise. This is a data-driven threshold, not a fixed constant, making it robust to the specific choice of ε and W. Features with lifespan ≥ δ are retained for Wasserstein distance computation; the threshold is reported as a hyperparameter alongside ε and W.

### 4.4 FDR computation protocol
The primary hypothesis claims a lower False Discovery Rate than the baseline detectors. FDR is computed as follows:

**Decision rule**: An alarm is raised at time t if SMFI(t) exceeds a detection threshold θ, chosen to fix the false-positive rate at 5% on the clean (non-injected) Tier 1 windows via the Neyman-Pearson construction — i.e., θ is the 95th percentile of the SMFI signal distribution on non-injected held-out data. All four baseline detectors (VPIN, OFI, quote-to-trade ratio, structural baseline) are thresholded by the identical false-positive-rate criterion on the same held-out windows, ensuring a fair comparison.

**Detection window**: An injection event is considered *detected* if at least one alarm falls within a window of [−5s, +30s] relative to the injection start time (i.e., detection is credited up to 5 seconds early and up to 30 seconds late). The 30-second window matches the enforcement-standard response window used in Do & Putniņš (2023). The 5-second pre-detection allowance accounts for the fact that structural LOB changes begin before the first injected cancel is formally logged.

**Multiple-testing correction — p-value definition (M4)**: Since FDR is evaluated across a corpus of N_inj injection events (N_inj ≥ 200, Tier 1 budget), and each injection event constitutes a hypothesis test, the Benjamini-Hochberg procedure at target FDR level q = 0.05 is applied to the ranked p-values of the per-event detection statistics.

**Per-event p-value definition (M4)**: The p-value for each injection event i is computed as a **one-sided empirical permutation p-value**: the fraction of clean (non-injected), held-out 35-second windows whose peak SMFI value strictly exceeds the maximum SMFI value observed in the injection window [−5s, +30s] surrounding event i. Formally: p_i = #{clean windows j : max_{t ∈ [−5s,+30s]} SMFI_j(t) > max_{t ∈ [−5s,+30s]} SMFI_i(t)} / N_clean, where N_clean is the total count of clean held-out 35-second windows. This is a distribution-free, non-parametric p-value under the empirical null of no injection. BH correction is applied to the ranked vector {p_1, ..., p_{N_inj}}; the headline FDR statistic reported is the empirical false-discovery proportion on the BH-controlled discovery list. Lead-time is computed only on true-discovery events, reported as median and interquartile range.

---

## 5. Cross-Asset Systemic Extension: Bounded Multiparameter Zigzag Persistence

### 5.1 Motivation and scoping decision

As designed above, SMFI is single-instrument — which sits uneasily with "Systemic" in its own title, and none of the papers surfaced in the prior-art search (Section 2) perform topological manipulation detection across correlated instruments jointly. Real manipulation campaigns are sometimes cross-market (moving one instrument's displayed liquidity to profit on a correlated instrument, or coordinated layering across multiple related tickers) — a pattern single-asset SMFI cannot see by construction, since it only ever looks at one order book at a time.

**Scope decision, stated explicitly**: this extension is scoped to **cross-asset, same-venue** detection (a small cluster of correlated NASDAQ tickers), not cross-venue detection (e.g., NASDAQ vs. a futures exchange), because LOBSTER's data coverage is NASDAQ-only. Cross-venue extension is noted as future work in Section 8, not claimed here.

### 5.2 Construction: a zigzag of asset-coupling complexes

At each time window t, a simplicial complex is built over a fixed cluster of k correlated tickers (vertices = tickers), with an edge between two tickers when the signature-distance between their log-signature-embedded LOB paths (Section 4.1) in that window falls below a threshold ε — i.e., their liquidity dynamics are "topologically coupled" in that window. As t advances, edges appear and disappear non-monotonically (two tickers can decouple and later re-couple), which is exactly the setting zigzag persistence — rather than ordinary persistence, which requires monotonic nesting — is built for. The resulting time-indexed sequence of asset-coupling complexes is itself run through zigzag persistence, producing a second-order fragility signal: rather than tracking liquidity cavities *within* one book, it tracks the birth and death of *coupling structure between* books. A sudden, anomalous coupling event (two normally-uncorrelated tickers suddenly showing signature-level coupling, concurrent with a Tier-1-style fragility spike in one of them) is the systemic-manipulation signature this extension is built to surface.

This is a genuine two-parameter (time × inter-asset coupling threshold ε) filtration problem, which places it in the domain of multidimensional/multiparameter persistence (Carlsson & Zomorodian, 2009).

**Cross-paper calibration note (ε and β_ij)**: The coupling threshold ε used here (edge inclusion when signature-distance < ε) and the coupling weight β_ij in the companion CT-ESN paper's coupled-reservoir construction (Section 6.2 of the companion paper, calibrated to match the empirical signature-distance correlation) measure the same underlying cross-asset relationship — how similar two instruments' log-signature-embedded LOB paths are in a given window. The two quantities are related as follows: ε is a binary edge-inclusion threshold (is the signature distance below ε?), while β_ij is a continuous coupling weight (how strongly should reservoir i respond to reservoir j's state, weighted by their historical signature-distance correlation). Both are calibrated on the same real LOB data for the same k=3 cluster over the same training window, ensuring that the "coupling strength" the generator (CT-ESN) was calibrated to reproduce is the same structure the detector (SMFI cross-asset extension) is built to detect.

### 5.3 Feasibility bound, stated honestly

Carlsson & Zomorodian's own foundational result on multidimensional persistence establishes that exact computation is EXPSPACE-complete in the worst case, with standard algorithms scaling doubly-exponentially — a materially harder computational regime than the single-parameter zigzag persistence used in Section 4. This is not a detail to gloss over: it is the reason this extension is deliberately **bounded** rather than run at the same scale as Tier 1:

- **Cluster size fixed at k = 3 tickers** (not an arbitrary large basket), keeping the coupling complex small enough that even doubly-exponential worst-case behavior remains tractable in practice.
- **The rank invariant** (Carlsson & Zomorodian's proposed discrete invariant for multidimensional persistence, valid and provably complete in the one-dimensional case) is used as the working summary statistic in place of attempting a full multidimensional barcode decomposition, which is not guaranteed to exist as a complete invariant beyond one dimension.
- **Evaluated cluster**: the free LOBSTER sample tickers (AAPL, AMZN, GOOG, INTC, MSFT) already form a natural, highly correlated mega-cap technology cluster (e.g. AAPL, AMZN, MSFT), used as the complete evaluation cluster for the cross-asset case study without requiring paid data.
- **This extension is explicitly reported as a bounded case study**, not a fully-powered statistical claim on par with Tier 1 — the secondary hypothesis in Section 1 is worded to reflect that.

### 5.4 Compute budget for this extension

Bounded separately from Tier 1 and Tier 3: an additional **20 GPU-hours**, applied to a small number of 3-ticker clusters over a reduced date range (targeting on the order of 20–30 ticker-windows per cluster, sufficient for a case-study-level demonstration, not a corpus-level FDR statistic). Total project compute across all three components is therefore **100 (Tier 1) + 20 (Tier 3 / CAEP) + 20 (cross-asset extension) = 140 GPU-hours**, an explicit increase over the previous draft's 100-hour figure, stated here rather than absorbed silently.

---

## 6. Datasets: Exact, Usable Sources for This Research

### 6.1 Primary corpus — LOBSTER (Public Academic Sample) + FI-2010 (merged, D1)
The Tier 1 injection corpus requires N_inj ≥ 200 independent, non-overlapping 35-second injection windows. The LOBSTER public sample covers approximately 1–2 trading days per ticker, which alone may not provide sufficient temporal bandwidth. **FI-2010 (10 full trading days, ~4.5M events across 5 Helsinki tickers) is therefore merged directly into the Tier 1 injection corpus** to mathematically guarantee the required window count without paid data. LOBSTER and FI-2010 windows are processed through identical normalization and injection pipelines but are never mixed within the same evaluation window; source identity (LOBSTER vs. FI-2010) is recorded as a covariate in all results. FI-2010 additionally serves as the cross-market / European equity generalization check (its held-out evaluation windows are disjoint from the Tier 1 training split).

### 6.2 Cross-market generalization check — FI-2010
FI-2010 is simultaneously part of the merged Tier 1 corpus (Section 6.1) and the cross-market generalization check. Its held-out evaluation windows (those not used for Tier 1 injection training) confirm single-asset SMFI results are not NASDAQ-specific.

### 6.3 Cross-asset-class stress test — Binance public order book API
Free, API-key-only; provides high-frequency crypto microstructure and candidate source for a LUNA-crash-based Tier 2 event. **D2 — Binance snapshot adapter**: Binance delivers order book state as discrete depth-ladder snapshots, structurally different from LOBSTER's continuous event-stream. A dedicated `SnapshotToEventAdapter` normalizes these into the shared tensor schema. Binance data is kept strictly isolated to Tier 2/3 stress tests and is **never mixed** into the Tier 1 continuous-message corpus.

### 6.4 Summary table

| Dataset | Cost | Scope | Role in this paper |
|---|---|---|---|
| LOBSTER (Public Sample) | $0 | AAPL, AMZN, GOOG, INTC, MSFT | **Tier 1 Corpus (primary)**: clean baseline windows; k=3 cross-asset cluster (AAPL-AMZN-MSFT) |
| **FI-2010 Benchmark** | $0 | 5 Helsinki tickers, 10 days | **Tier 1 Corpus (merged)** to guarantee N_inj ≥ 200; cross-market generalization check |
| Binance Public API | $0 | Spot pairs, live + historical | Tier 2/3 stress test (via SnapshotToEventAdapter; isolated from Tier 1) |
| Public Regulatory Exhibits | $0 | CFTC/SEC releases (Coscia, Flash Crash) | Tier 2 qualitative real-world sanity check |

**Data budget constraint**: Strictly $0 paid data. All evaluations, injections, and cross-asset runs are executed exclusively on verified free, public open-access data.

---

## 7. Full Pipeline

1. Raw ingestion: LOBSTER `message`/`orderbook` pairs (dynamic depth-level column detection) + FI-2010 (merged corpus, identical normalization); Binance data via `SnapshotToEventAdapter` (Tier 2/3 only, kept strictly separate from Tier 1 corpus).
2. Event filtering & alignment (four core event types).
3. Single-asset path construction (Section 4).
4. **Tier 1 injection**: parameterized spoofing/layering (Cartea/Jaimungal/Wang and Tao et al.), applied independently per ticker across both LOBSTER and FI-2010 windows. **A2 — Coordinated-Injection Subset**: a dedicated sub-corpus of ~30 injection events (out of N_inj ≥ 200 total) applies spoofing parameters **simultaneously** across all 3 tickers of the cross-asset cluster (AAPL-AMZN-MSFT) in the same market window, simulating coordinated cross-asset manipulation. This coordinated subset is the empirical basis for testing the secondary hypothesis (Section 1); it is labeled and tracked separately from the independent-injection majority.
5. Log-signature extraction (N=4; W grid-searched across {5, 10, 20, 50}, calibrated jointly with ε on clean calibration split).
6. Simplicial complex construction (single-asset, Vietoris-Rips at radius ε).
7. Zigzag persistence (Dionysus2 primary, GUDHI cross-check). **Agreement criterion and halting protocol**: persistence diagrams from both libraries are required to agree within ε_pd = 0.01 (in the 2-Wasserstein metric between diagrams) on all generators with lifespan > δ, evaluated on a held-out validation set of 50 clean + 50 injected windows before any Tier 1 results are computed. Any isolated window with discrepancy > ε_pd is flagged and excluded; its exclusion rate is documented. **Strict halting condition**: if more than 5% of validation windows exceed ε_pd, execution halts immediately to investigate systemic library discrepancies.
8. SMFI signal computation (2-Wasserstein distance between consecutive diagrams, lifespan filter δ).
9. Baseline computation: VPIN, OFI, quote-to-trade ratio, **plus one structural 2024–2025 baseline** (Zheng's motif-GNN or Fabre & Challet's Hawkes model).
10. Detection & FDR/lead-time scoring (Tier 1): one-sided permutation p-values per event (Section 4.4) → Benjamini-Hochberg FDR correction at q = 0.05.
11. Tier 2 replay, expanded case set (Section 3.2).
12. **Cross-asset extension**: build per-window asset-coupling complexes over the 3-ticker cluster (Section 5.2); run zigzag persistence over the coupling-complex sequence; compute rank-invariant summary. For the coordinated-injection subset (Step 4), compare coupling-anomaly timing against independently-run single-asset Tier 1 signals to test whether the cross-asset signal precedes or outperforms single-asset detection (the secondary hypothesis).
13. **CAEP / Tier 3**: instantiate the generalized protocol (Section 3.4) with D = SMFI, G = CT-ESN, M = the Cartea/Tao-grounded manipulation floor; run the constrained adversarial loop; report the three CAEP outputs.

---

## 8. Limitations, Stated Explicitly

- Tier 1's FDR statistic depends on the realism of the parameterized injection family; it is now grounded in two independent analytical spoofing models (Cartea/Jaimungal/Wang; Tao et al.) rather than an arbitrary range, but remains a synthetic construction, not real labeled data.
- Tier 2 remains qualitative regardless of how many cases are added from Do & Putniņš's sample; it is a sanity check, not a source of statistical power for the headline FDR claim.
- The cross-asset extension (Section 5) is a bounded case study, not a fully-powered statistical result, because multidimensional persistence computation is EXPSPACE-complete in the worst case — this is a stated scope boundary, not an oversight. Cluster size (k=3) and the rank-invariant approximation are the specific mitigations; a full multi-asset, arbitrary-cluster-size version of this idea is future work, not a claim of this paper.
- The cross-asset extension is scoped to same-venue (NASDAQ) correlated tickers; true cross-venue manipulation (e.g., futures-to-equity) is out of scope given LOBSTER's coverage.
- **Cross-asset CAEP stress test is an explicit non-goal**: While single-asset CAEP (Section 3.4) pairs single-asset SMFI against single-asset CT-ESN, running a fully joint adversarial loop on the cross-asset system (i.e., optimizing a coupled-reservoir adversarial generator against the multiparameter zigzag coupling detector) is omitted from this work to respect strict compute boundaries (140 GPU-hours total). This full-circle systemic adversarial benchmark is designated as a direct next step for future work.
- CAEP's reported "constraint-violation rate" and "compute cost per adversarial search" are properties of this specific instantiation (SMFI + CT-ESN); generalizing CAEP's actual utility to other detector/generator pairs is asserted as a methodological contribution but not empirically demonstrated beyond the one instance in this paper.
- The added structural baseline (Zheng or Fabre & Challet) depends on reference code being available at implementation time; if neither is available, this reverts to the volume-based baseline set only, and that reversion should be stated in the final paper rather than silently omitted.

---

## References

- Andersen, T. G., & Bondarenko, O. (2014). VPIN and the Flash Crash. *Journal of Financial Markets*, 17, 1–46.
- Briola, A., Bartolucci, S., & Aste, T. (2024). HLOB: Learning the higher-order structure of limit order books.
- Carlsson, G., de Silva, V., & Morozov, D. (2009). Zigzag persistent homology and real-valued functions. *Proceedings of the 25th Annual Symposium on Computational Geometry (SoCG)*, 247–256.
- Carlsson, G., & Zomorodian, A. (2009). The theory of multidimensional persistence. *Discrete & Computational Geometry*, 42(1), 71–93.
- Cartea, Á., Chang, P., & García-Arenas, M. (2023). Spoofing and manipulating order books with learning algorithms.
- Cartea, Á., Jaimungal, S., & Wang, Y. (2020). Spoofing and price manipulation in order-driven markets.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103–120.
- Do, H. X., & Putniņš, T. J. (2023). Detecting layering and spoofing in markets.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457–1493.
- Esteban, C., Hyland, S. L., & Rätsch, G. (2017). Real-valued (medical) time series generation with recurrent conditional GANs. *arXiv:1706.02633*.
- Fabre, B., & Challet, D. (2025). A Hawkes-process model of order-book spoofability.
- Gidea, M., & Katz, Y. (2018). Topological data analysis of financial time series: Landscapes of crashes. *Physica A*, 491, 820–834.
- Kularatnam, K., & Stathaki, T. (2024). Weakly-supervised temporal convolutional detection of order-book spoofing.
- Li, K., Polukarova, M., & Ventre, C. (2023). Statistical-physics detection of market manipulation, validated on the LUNA crash.
- Lin, T., & Yang, S. (2025). Cascaded contrastive representation learning for multilevel order-book manipulation detection.
- Lyons, T. (1998). Differential equations driven by rough signals. *Revista Matemática Iberoamericana*, 14(2), 215–310.
- Maria, C., & Oudot, S. Y. (2014). Zigzag persistence via reflections and transpositions. *Proceedings of SODA*.
- Milosavljević, N., Morozov, D., & Skraba, P. (2011). Zigzag persistent homology in matrix multiplication time. *Proceedings of SoCG*.
- Ni, H., Szpruch, L., Wiese, M., Liao, S., & Xiao, B. (2020). Conditional Sig-Wasserstein GANs for time series generation.
- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark dataset for mid-price forecasting of limit order book data. *Journal of Forecasting*, 37(8), 852–866.
- Souto, H. G., et al. (2024). FinTDA: Topological data analysis for financial time series.
- Tao, X., Day, A., Ling, L., & Drapeau, S. (2020). On detecting spoofing strategies in high-frequency trading.
- Turiel, J. D., Barucca, P., & Aste, T. (2022). Simplicial persistence of market correlation structure as a systemic risk indicator.
- United States v. Coscia, 866 F.3d 782 (7th Cir. 2017).
- Zheng, Y. (2025). Temporal motif mining and motif-aware graph neural networks for real-time spoofing detection.

---

*Final revision. Novelty framing narrowed and grounded against the prior-art search in Section 2; genuine novelty extended via the cross-asset multiparameter extension (Section 5) and the generalized adversarial evaluation protocol (Section 3.4), both scoped and budgeted explicitly rather than added without accounting for their cost.*