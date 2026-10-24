# Ordinary for This Moment? Generator-Calibrated Surveillance of Limit Order Books with False-Discovery Control

**Arshad Khatib** — B.Tech in Artificial Intelligence (5th semester), SVNIT Surat
Research proposal, 21 October 2026.

---

## Abstract

Market surveillance flags order flow that looks unusual. The hard part is *unusual compared with what*. A burst of quotes is routine at a busy moment and suspicious at a quiet one, so a detector calibrated on pooled "normal" windows misses manipulation in calm periods and drowns in false alarms in busy ones. We propose to answer "is this window ordinary *for this moment*?" with a conditional null model: a level-1 order-book generator, developed and validated in our companion project, that simulates how the next 100 events could have unfolded given the actual preceding events and the market's slow activity state. A window's tail probability under those simulations ranks how surprising it is. Split-conformal calibration on real clean windows then turns that ranking into p-values that are valid whatever the generator gets wrong, and Benjamini–Hochberg controls the false-discovery rate. Multilevel splitting, already validated against brute force on this generator, resolves deep tails that plain simulation cannot reach. We test four things on a 30-day, six-symbol crypto futures panel (with one day of NASDAQ data as a secondary set), using labelled injections of three manipulation footprints visible at level 1 (touch spoofing, quote stuffing, momentum ignition):
- whether the p-values are valid;
- whether conditioning on the moment buys detection power over the same statistics with an unconditional null;
- whether the approach detects a manipulation pattern it was never shown better than a supervised detector trained on the other patterns;
- whether a topological statistic adds anything over a nearest-neighbour one.

We also measure an *evasion frontier*: how much price effect a manipulator can obtain while staying below a given detection rate.

---

## 1. Why this study, and how it continues the generator work

### 1.1 What the companion project has established

These are facts from the generator repository (orderflow-noise-models). Every number below comes from its result files; none is re-estimated here.

- **The generator.** It is a closed-form, level-1 event generator: a VAR mean readout, a log-scale readout that includes slow activity averages (half-lives of 10, 100 and 1,000 events), and residuals resampled from the 64 nearest of the 30,000 most recent real moments. It is called `slow-s|W30000`. Its features per event are mid-price move, spread, log best-bid size, log best-ask size and log inter-event time. The configuration was fixed on 2026-10-14 before any test-split run.
- **Test-split confirmation** (job 32358, 10 datasets × 5 seeds, pre-registered arm):
  - signature-MMD two-sample test passed in 31/50 runs (validation: 38/50);
  - median rare-event calibration factor 1.34; 80% of event levels within 2×; worst 6.4×;
  - by comparison, the pilot-4 analog sampler passed 12/50 and an iid-residual sampler 0/50.
- **Multilevel splitting.** Splitting agrees with brute force on the generator's own law: |z| < 2 on 93–98% of checked levels. Its estimates stay usable to about 1e-6.
- **Known weaknesses that matter here:**
  - FI-2010 never passes;
  - bursts are 2.3–2.7× off on AAPL/AMZN;
  - spread extremes are about 3× off on DOGE/LTC;
  - on LOBSTER, queue sizes jitter at almost every event (§6.1, measured for this proposal).

### 1.2 What this study adds

The generator project asked whether synthetic order flow is realistic. This study uses that realism for something else: as the null model of a statistical test. The roles divide cleanly:

| Component | Role | What it is responsible for |
|---|---|---|
| Generator (frozen) | ranks windows by how surprising they are given their context | power |
| Multilevel splitting | resolves the ranking in the far tail | resolution |
| Conformal calibration on real clean windows | turns ranks into p-values | validity, regardless of generator errors |
| Benjamini–Hochberg | chooses which windows to flag | false-discovery control |

If the generator is wrong, the test loses power but stays valid. That is the right failure mode for a surveillance tool.

The work also flows back to the generator. Using the generator as a null exposes defects that its own fidelity tests did not, such as the LOBSTER queue-size jitter. Any fix is validated and reported as a new generator version (§6.1).

### 1.3 Novelty, stated narrowly

Each piece exists separately:
- conformal outlier detection with FDR control (Bates et al., 2023);
- conditional LOB generators used as reactive simulators (Coletta et al., 2023);
- agent-based simulation of spoofing (Wang and Wellman);
- supervised and Hawkes-based spoofing detectors (Kularatnam and Stathaki, 2024; Fabre and Challet, 2025; Lin and Yang, 2025).

In a literature search on 2026-10-21 we found no work that uses a fitted conditional order-book generator as the null distribution of a surveillance test, calibrates it conformally, and evaluates FDR and power against labelled injections. The claim is that combination and its evaluation. We do not claim novelty for any single ingredient, and the search will be repeated before submission.

---

## 2. Hypotheses (pre-registered)

All decisions below are fixed in `config/preregistration.yaml` before the test split is touched (§5).

**Common settings.**
- *Windows:* 100-event context and 100-event observation window; non-overlapping windows for all FDR analyses.
- *FDR level:* q = 0.10.
- *Injections:* 5% of test windows are injected.
- *Primary corpus:* the Binance panel (6 symbols).
- *Primary metric:* power at q = 0.10, averaged over the pre-specified grid of injection families and strengths.

**H1 — validity.** Conformal-calibrated p-values control FDR.
- *Measure:* the empirical FDR on injected test sets, averaged over 3 injection seeds.
- *Pass:* FDR ≤ 0.10 + 2 standard errors on each symbol.
- *Also reported:* on clean test windows, the discovery count and a QQ plot of the p-values.
- *Diagnostic, not a pass criterion:* we also report the uncalibrated generator p-values' miscalibration in the tail (splitting estimates against real exceedance frequencies). This measures how far the generator alone can be trusted.

**H2 — conditioning buys power (primary).** At q = 0.10, the generator-conditional score detects more injected windows than an otherwise identical pipeline whose null is the unconditional distribution of the same statistics over clean calibration windows.
- *Test:* a window-level paired comparison (McNemar test, pooled over symbols) plus a symbol-level sign test (6 symbols).
- *Supported* if both give p < 0.05 in favour of conditioning.
- *Refuted* if conditioning has lower power on 4 or more symbols.

**H3 — unseen patterns.** For each manipulation family in turn, a supervised detector is trained on injected validation windows of the *other* two families. Its scores are calibrated conformally exactly like ours. The generator-conditional detector, which never sees any injection, is compared with it on the held-out family.
- *Supported* if the generator-conditional detector matches or beats the supervised one on at least 2 of 3 held-out families.
- We report the supervised detector on *seen* families too, as an upper reference.

**H4 — topology (secondary).** Adding an H1-persistence statistic of window log-signature clouds to the statistic panel raises power, compared with adding its H0 / minimum-spanning-tree counterpart on the same clouds. The two arms differ in one statistic.
- *Test:* paired, as in H2.
- A null result is reported as such.

**H5 — cross-asset (exploratory).** Coordinated injections across at least 3 symbols, each too weak to be detected on its own symbol, are detected more often by combining per-symbol p-values across symbols than by any single symbol.

**M1 — evasion frontier (measurement, not a hypothesis).** For touch spoofing and momentum ignition, we map detection power and the manipulator's price effect over a grid of parameters, and report the largest effect obtainable at power ≤ 20% and ≤ 50%. The price effect requires the generator's *reactive* response, so M1 is reported only if the realism check R (§3.8) passes.

---

> **Revision, 23 October 2026 (Stage-0 redesign, validation only).** The primary measure of H2 is now power at a
> fixed false-alarm rate (share of injected windows with conformal p <= 0.05; 0.01 secondary), averaged over a grid of
> injection strengths, instead of BH power at 5% prevalence; BH power and FDR remain as secondary measures. Reason: the
> first Stage-0 pilot showed that BH could not reject anything at realistic calibration sizes and signal strengths.
> The statistic panel gains `queue_roundtrip`, and the generator null gets more continuations for extreme windows so its
> resolution matches the unconditional null's. Details and the go/no-go rule: `results/stage0_redesign.md`.

## 3. Method

### 3.1 Data

| Set | Content | Split | Role |
|---|---|---|---|
| **Binance USD-M futures** (BTC, ETH, SOL, XRP, DOGE, LTC) | `bookTicker` level-1 events, 2024-02-01..03-01, the first 100k events per day after 14:00 UTC; ~1,000 non-overlapping windows per symbol-day | by day 18 / 6 / 6 (as in the generator repo) | **primary**: all hypotheses |
| Binance `aggTrades`, same symbols and days | trades (free, data.binance.vision) | same | trade-based baselines only (B3); downloaded in Stage 1 |
| **LOBSTER** AAPL, AMZN, MSFT | one full day, 10 levels, message stream; ~450 non-overlapping windows per split | chronological 70 / 15 / 15 | **secondary**: H1/H2 only, and only if generator fix G1 passes (§6.1) |
| FI-2010 | — | — | **excluded**: no timestamps or messages, and the generator never passes on it |

The DS pipeline needs event timestamps alongside the features, for wall-clock delays and for H5. `binance_data.py` already reads `transaction_time`; the feature cache must keep it.

### 3.2 The conditional null

For a window starting at stream position s, the generator is started from the real context `S[s-p:s]` and the real slow-activity state at s−1. It uses its recent pool, which ends before s, so there is no look-ahead. It then simulates M continuations of 100 events (`lobsurv.null_model.GeneratorNull`).

Injections are applied only to the observed window, never to the stream the generator reads. One set of simulations per window therefore serves every injection family and strength. This is the main cost saving.

### 3.3 Statistics panel

Each statistic is a running score whose maximum over the window is the window statistic (`lobsurv/statistics.py`). That is also the event form that multilevel splitting estimates.

| Statistic | Footprint it targets |
|---|---|
| `range` | range of cumulative mid-price |
| `n_moves` | number of mid-price moves |
| `max_spread` | liquidity withdrawal |
| `queue_swing` | largest change of best-queue imbalance relative to the end of the context: touch spoofing, which shifts displayed imbalance (the mechanism of Cartea, Jaimungal and Wang, 2020) |
| `queue_jump` | largest one-event change of a best queue at events that leave both best prices unchanged: the add and the cancel of a non-bona-fide order. Price-changing events are excluded, because a new queue taking over a level routinely jumps by 20× |
| `n_fast` | events faster than the training 5th percentile of inter-event time: quote stuffing |
| `queue_roundtrip` (Stage 1) | the add/cancel pair: a same-price jump up followed within Δ events by a jump down of similar size |

For each window and statistic j:
1. The generator p-value is p_j = (1 + #{simulated ≥ observed}) / (M + 1).
2. The p_j are combined with the Cauchy combination (Liu and Xie, 2020), which stays valid under the strong dependence between statistics.
3. The window score is g = −log p_comb.
4. For windows with p_comb at the resolution floor of the screening run, M is raised or multilevel splitting refines the ranking.

### 3.4 Calibration and discovery

- **Calibration windows:** for each symbol, the clean non-overlapping windows of the last 3 validation days (about 3,000).
- **Test windows:** the 6 test days (about 6,000 per symbol).
- **p-values:** conformal, p = (1 + #{calibration scores ≥ test score}) / (n_cal + 1), followed by BH at q = 0.10 within each symbol. Conformal p-values are positively dependent in the way BH needs, so BH controls FDR (Bates et al., 2023).
- **Contamination:** unknown real anomalies in the calibration windows make p-values larger, so they make the test conservative, not invalid.
- **Resolution:** with n_cal ≈ 3,000 and m ≈ 6,000, BH can reject once about 20 windows are flagged. The roughly 300 injected windows per symbol leave room for that.
- **Main threat:** calibration and test windows are not exactly exchangeable across days. This is measured directly through H1's empirical FDR, and a sensitivity variant calibrates on the previous day only.
- **Detection delay:** sliding windows with stride 20 events and BY correction. Delay is measured from the injection's first event to the end of the first flagged window, in events and seconds. There is no "early credit".

### 3.5 Injection families (labelled test cases)

All sizes are in each symbol's own units, estimated on its training days: multiples of the median best-queue size, the inter-event-time quantiles, and the tick. No external "realistic" values are claimed. Positions are drawn at random among test windows (5% prevalence, 3 seeds); the injection starts 10–60 events into the window.

| Family | What is injected (level-1 view) | Grid |
|---|---|---|
| **F1 touch spoofing** | best-bid (or best-ask) size raised by V = k × median queue at an event that leaves prices unchanged; removed after τ events, or as soon as that best price changes (the order is then no longer at the touch) | k ∈ {2, 5, 10, 20}, τ ∈ {5, 20, 50} |
| **F2 quote stuffing** | n extra events inserted, with inter-event times at the 1st percentile, each a one-lot flicker of a best queue with prices unchanged; the window is the first 100 events after insertion | n ∈ {10, 25, 50} |
| **F3 momentum ignition** | m consecutive one-tick mid moves in one direction (ask side consumed), followed by a return to the starting level within r events | m ∈ {2, 3, 5}, r ∈ {10, 30} |

**Two modes:**
- *Passive replay:* only the manipulator's footprint is added; everything else is the real window. This is used for H1–H5.
- *Reactive:* after the injection, the rest of the window is generated by the generator from the manipulated history, which models the market's response. This is used for M1 only.

Layering at depth is invisible at level 1 and is out of scope.

### 3.6 Baselines

All baselines are calibrated with the same conformal procedure on the same windows, so every comparison is at equal FDR.

- **B1 unconditional panel:** the same statistics; null = their distribution over clean calibration windows. This is the H2 comparator.
- **B2 one-step surprise:** the sum over the window of squared standardised one-step residuals under the generator's own mean and scale readouts. It tests whether multi-step simulation is needed at all.
- **B3 classic indicators:**
  - order-flow imbalance (Cont, Kukanov and Stoikov, 2014), maximum absolute value over the window;
  - event-rate z-score;
  - quote-to-trade ratio (from `aggTrades`).

  VPIN is computed at its natural volume-bucket scale and shown as context only, because 100-event windows are far shorter than its buckets.
- **B4 unsupervised machine learning:** isolation forest and k-nearest-neighbour distance on window log-signatures.
- **B5 supervised:** gradient-boosted trees on window log-signatures plus the panel. It is trained on injected validation windows and used for H3.
- **Optional B6:** Fabre and Challet (2025) or Lin and Yang (2025), only if their code is public when Stage 1 starts. Its absence is stated, not hidden.

### 3.7 Topological statistic (H4)

- **Point cloud per window:** log-signatures, at depth 3, of 10-event sub-paths taken at stride 2. Channels: time, mid-price, spread and imbalance, z-scored with training statistics. That gives 30 dimensions and about 46 points per window.
- **Statistics:** total H1 persistence of the cloud's Rips filtration (`lobsurv/persistence.py`) versus the H0 / minimum-spanning-tree analogue (mean edge length).

H0 death times *are* the minimum-spanning-tree edge lengths (checked in `tests/test_persistence.py`), so only H1 can add genuinely topological information. Both statistics are computed identically on real and simulated windows.

### 3.8 Evasion frontier (M1) and the realism check R

**Realism check R.** On held-out real days, take natural shocks: events where best-queue imbalance jumps above its 99th percentile at an unchanged price. Compare the real mean mid-price response 10, 50 and 100 events later with the generator's response from the same contexts.
- R passes if the generator's mean lies within the real 95% confidence interval at all three horizons on at least 4 of 6 symbols.

**Frontier.** If R passes, for each grid point θ of F1 and F3 compute:
- the manipulator's effect E(θ): the mean favourable mid-price move over the window in reactive mode, minus the same quantity without injection;
- the detection power D(θ) at q = 0.10.

We report the frontier max{E(θ) : D(θ) ≤ β} for β ∈ {0.2, 0.5}. If R fails, M1 is reported only as footprint size versus power, and we say that the price effect is unmeasured.

### 3.9 Cross-asset (H5)

- **Bins:** wall-clock bins of 5 s.
- **Per-symbol bin p-value:** the minimum p over the windows overlapping the bin, Bonferroni-adjusted by their count.
- **Combination:** Cauchy combination across the 6 symbols, then BH over bins.
- **Coordinated injections:** F1 or F3 on at least 3 symbols within the same 5 s. The per-symbol strength is chosen on validation so that single-symbol power is about 20%.
- **Comparator:** the best single symbol.

---

## 4. Controls

- **Generator-only negative control:** run the whole pipeline on windows *generated* by the generator, so every window is null. The discovery rate must be consistent with no true discoveries.
- **Clean-test control:** clean test windows with no injections. Their discoveries are genuine anomalies or generator misfit, and the two cannot be separated. We report the count and a manual categorisation of the top 50 (e.g. liquidation cascade, spread blow-out, data gap), with timestamps. No discovery is called manipulation.
- **Same-statistic control:** B1 differs from the method only in its null, so H2 isolates the effect of conditioning.
- **Generator-version control:** every result file stores the generator commit and its count of uncommitted files (`generator_provenance()`).

---

## 5. Protocol and pre-registration

1. **Development (Stages 0–1)** uses validation days only: injection code, statistic choice including `queue_roundtrip`, M, the screening rule and the B5 features.
2. **Freeze.** `config/preregistration.yaml` is committed before the test run. It holds the panel, M, q, calibration days, grids, seeds, the primary metric and every pass/fail rule in §2.
3. **One test run** with the frozen configuration. Anything changed afterwards is reported as exploratory, together with the frozen result.
4. **Disclosure.** The generator's test split was used once before, to confirm the generator. This study does not select or tune the generator on it, and says so. If generator fix G1 is adopted, the paper reports it as a new generator version, selected on validation.

---

## 6. Feasibility

### 6.1 Measured on 2026-10-21 (Stage 0 preview)

**Cost.** One window's 1,000 continuations take about 3 s on AAPL with 4 neighbour-search workers. Fitting the generator takes a few seconds per dataset from cache.

**Queue-size dynamics.** We looked at events that leave both best prices unchanged, over 20 test windows:

| Dataset | Share with unchanged best queues (real / generated) | 90th percentile of \|Δ log queue\| (real / generated) |
|---|---|---|
| AAPL | **0.89 / 0.04** | 0.06 / 0.40 |
| BTCUSDT | 0.00 / 0.00 | 0.10 / 0.11 |

On Binance, every `bookTicker` event changes the touch, and the generator matches. On LOBSTER, most events happen deeper in the book, so real best queues usually stay the same, while the generator jitters them at almost every event. That makes the null far too wide for queue-based statistics: a 20× spoof order scored p ≈ 0.14 on AAPL.

**G1, the generator fix needed before LOBSTER can be used:** draw queue-size changes from the chosen neighbour's own change, as the existing `inc` variants already do for mid-price and spread. That keeps the large mass at "no change".
- *Pass:* the unchanged share is within 0.05 of real, and the validation MMD pass rate does not drop.
- *If G1 fails,* LOBSTER is dropped and the study is Binance-only.

### 6.2 Compute (CPU-bound)

**Windows:** per symbol, about 3,000 calibration plus 6,000 test, so about 54,000 windows across 6 symbols.

**Screening at M = 200**, with refinement at M = 2,000 for the top 5%:

| Step | Cost |
|---|---|
| Screening | 54,000 × ~0.6 s ≈ 9 h on 4 cores |
| Refinement | 2,700 × ~6 s ≈ 4.5 h |
| Total | ≈ 55 core-hours |
| H4 persistence on simulated paths (subsample of 5,000 windows × M = 200) | ≈ 20 core-hours |

That is well within one cluster job of 16–64 CPUs.

**GPU.** Porting the neighbour search to the GPU (planned in the generator repo) would cut this further; it is not required.

---

## 7. Work plan

| Stage | Content | Exit criterion |
|---|---|---|
| 0 (week 1) | bridge to the generator (done); cost measurement (done); timestamps in the cache; QQ of generator p-values on clean validation windows; G1 on LOBSTER | G1 decision; the method has power above B1 on a small validation pilot. If not, the study becomes a documented negative result |
| 1 (weeks 2–3) | injectors F1–F3 with unit tests; baselines B1–B5; `aggTrades` download; `queue_roundtrip`; validation tuning | all components tested; pre-registration written |
| 2 (week 4) | freeze; one cluster test run (H1–H3) | result files with provenance |
| 3 (week 5) | H4, H5, R and M1 | — |
| 4 (week 6) | write-up; repeat the literature search; check every reference | — |

---

## 8. Limitations

- **Injected labels are synthetic.** Power on real manipulation is unknown. No labelled real manipulation is available at $0, and the study does not claim to detect any.
- **Level 1 only.** Layering at depth, cross-venue manipulation and trade-based schemes are out of scope.
- **Validity rests on approximate exchangeability** of calibration and test windows across days. It is measured (H1), not assumed.
- **Generator errors cost power, not validity** (conformal calibration). Known generator errors (bursts on AAPL/AMZN, spread extremes on DOGE/LTC, FI-2010) mark where power may be lost.
- **Reactive responses are learned correlations, not causal effects.** M1 is conditional on R and is a model-based estimate.
- **Six crypto symbols over one month** limit generality. One LOBSTER day adds a second market only if G1 passes.

---

## 9. References

Status: **V** = verified on 2026-10-21 (title, authors and venue seen); **K** = standard reference, to be re-checked at Stage 4.

| Ref | Status |
|---|---|
| Bates, S., Candès, E., Lei, L., Romano, Y., Sesia, M. (2023). Testing for outliers with conformal p-values. *Annals of Statistics* 51(1). | V |
| Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. B* 57(1), 289–300. | K |
| Benjamini, Y., Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics* 29(4), 1165–1188. | K |
| Bréhier, C.-E., Gazeau, M., Goudenège, L., Lelièvre, T., Rousset, M. (2016). Unbiasedness of some generalized adaptive multilevel splitting algorithms. *Annals of Applied Probability* 26(6). | K (used in the generator repo) |
| Briola, A., Bartolucci, S., Aste, T. (2025). HLOB — Information persistence and structure in limit order books. *Expert Systems with Applications* 266, 126078. arXiv:2405.18938. | V |
| Cartea, Á., Chang, P., García-Arenas, G. Spoofing and manipulating order books with learning algorithms. Working paper, Oxford-Man Institute. | V (SSRN ID to confirm) |
| Cartea, Á., Jaimungal, S., Wang, Y. (2020). Spoofing and price manipulation in order-driven markets. *Applied Mathematical Finance* 27(1–2), 67–98. | V |
| Coletta, A., et al. (2023). Conditional generators for limit order book environments: explainability, challenges, and robustness. arXiv:2306.12806. | V |
| Cont, R., Kukanov, A., Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1), 47–88. | K |
| Easley, D., López de Prado, M., O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies* 25(5), 1457–1493. | K |
| Fabre, B., Challet, D. (2025). Learning the spoofability of limit order books with interpretable probabilistic neural networks. arXiv:2504.15908. | V |
| Gidea, M., Katz, Y. (2018). Topological data analysis of financial time series: landscapes of crashes. *Physica A* 491, 820–834. | K |
| Kularatnam, K., Stathaki, T. (2024). Detecting and triaging spoofing using temporal convolutional networks. arXiv:2403.13429 (AAAI 2024 workshop). | V |
| Lin, Y., Yang, P. (2025). Detecting multilevel manipulation from limit order book via cascaded contrastive representation learning. arXiv:2508.17086. | V |
| Liu, Y., Xie, J. (2020). Cauchy combination test. *Journal of the American Statistical Association* 115(529), 393–402. | K |
| Tao, X., Day, A., Ling, L., Drapeau, S. (2022). On detecting spoofing strategies in high-frequency trading. *Quantitative Finance* 22(8), 1405–1425. | V |
| Turiel, J. D., Barucca, P., Aste, T. (2022). Simplicial persistence of financial markets: filtering, generative processes and structural risk. *Entropy*. | V |
| Wang, X., Wellman, M. P. Agent-based modelling of spoofing in limit order markets. (Venue and year to confirm.) | to verify |
| Companion: Khatib, A., et al. Closed-form level-1 order-book generator (orderflow-noise-models repository; results in `benchmark_results/v2/`). | — |

Do not add a reference to this list without seeing the paper itself.
