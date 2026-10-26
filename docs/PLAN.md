# Plan (about 5–6 weeks of evenings)

Goal: a retail PD model on Lending Club loans, built and validated the way a bank's model risk
team would expect: a WoE/logistic scorecard as the benchmark, LightGBM as the challenger,
out-of-time validation, stability monitoring, and PD turned into expected loss with a stress
scenario. Deliverables: this repo (one-command reproduce), a model documentation pack in
validation-review style (`reports/model_doc/`), and a post on jurstadev.xyz.

## Data (checked 2026-10-26)
- Lending Club accepted loans 2007–2018Q4 (`accepted_2007_to_2018Q4.csv`, ~1.7 GB, ~2.26M rows,
  151 columns). Source: the Kaggle "wordsforthewise/lending-club" file, fetched from a CC0
  mirror on Hugging Face (`codesignal/lending-club-loan-accepted`) because no Kaggle
  credentials are set up. Check row count and columns against the Kaggle description; record
  the sha256 in `data/SOURCES.md`. Raw data stays out of git.
- Why not Home Credit: it has no application dates, so out-of-time testing and PSI across
  vintages (the core of this project) are impossible.

## Key design decisions
- **Population:** 36-month loans only, issued while every loan could still run to maturity
  before the data snapshot (issue date ≤ 2015-12). Every loan then has a final outcome, so
  recent vintages aren't biased towards early defaults. 60-month loans are out of scope (note
  in the doc as a limitation).
- **Target:** bad = Charged Off / Default (incl. "Does not meet the credit policy" variants);
  good = Fully Paid. Late/Current at snapshot are dropped (should be ~none after the filter;
  count and report them). This is a *lifetime* (36-month) PD; the doc also gives an
  approximate 12-month PD from the default timing (`last_pymnt_d`) for the Basel-style EL.
- **Features:** application-time fields only (the Zenodo "granting model" variable list is a
  cross-check). Everything set after origination (payments, recoveries, last FICO,
  hardship, settlement) is leakage and is excluded by an explicit allowlist, not a denylist.
- **Lending Club's grade and interest rate are not features.** They are LC's own model.
  They become a third benchmark: does our scorecard rank risk better than LC's grade did?
- **Samples (decided 2026-10-26):** development = issue 2012Q4–2013Q4, 115,603 loans
  (random 70/30 train/test inside it); out-of-time = 2014 (162,570) and 2015 (282,787)
  vintages, scored separately. LC added ~15 bureau fields (tot_cur_bal, bc_util, mort_acc,
  acc_open_past_24mths, ...) only from 2012Q4; 2007–2012Q3 lack them, so those vintages
  calibrate the stress scenario (grade-level default rates) and are not used for fitting. Thresholds, bins and
  hyperparameters are frozen before the OOT sample is touched.

## Phase 1: Data pipeline (week 1)
- [x] Download + checksum; `data/SOURCES.md`. 2,260,668 loans, matches Kaggle.
- [x] Load to parquet (`lcrisk ingest`, 30s)
- [x] Population filters + target; waterfall in `data/processed/waterfall.csv`:
      620,783 loans, bad rate 13.94%; only 147 dropped for no final outcome
- [x] Coverage by vintage (`lcrisk profile`) → dev window decision above
- Finding for validation: bad rate rises within *every* grade 2013 → 2015 (C 15.3% → 19.4%,
  E 23.3% → 32.8%). Expect OOT under-prediction; level drift is a headline result.
- [x] Feature allowlist, cleaning (emp_length, percent strings, dates → credit history
      length), missing-value treatment documented per field
- [x] Default rate by vintage and grade (sanity chart; 2007–08 should stand out)

## First fit (2026-10-26, `lcrisk fit`, ~3 min)
| Gini        | test  | OOT 2014 | OOT 2015 |
|-------------|-------|----------|----------|
| Scorecard   | 0.310 | 0.316    | 0.330    |
| LightGBM    | 0.352 | 0.354    | 0.363    |
| LC subgrade | 0.277 | 0.319    | 0.357    |
- Ranking holds out of time for both models (no Gini decay).
- Calibration drifts: mean PD ~12.9–13.3% vs observed 13.7% (2014) and 14.9% (2015).
- LC's own grade gets much sharper by 2015, catching LightGBM; it was trained on more data
  and saw the full bureau report. Worth a paragraph.
- LightGBM in-sample Gini 0.43 vs 0.35 test: overfit gap to discuss / regularise.

## Phase 2: Benchmark scorecard (week 2)
- [x] Monotonic WoE binning (own implementation, tested), IV per feature
- [x] Feature selection: IV ≥ 0.02, correlation pruning (|r| > 0.6), sign check → 10 features
- [x] VIF table for the doc
- [x] Missing bins under 0.5% of train score neutral (two features had a 1-loan missing bin)
- [x] Logistic regression on WoE; points scaling (600 at 50:1 odds, PDO 20); scorecard table
- [x] Rating grades (master scale R1–R8) from score bands

## Phase 3: Challenger (week 3)
- [x] LightGBM on raw features, monotone constraints where the business logic is clear
- [x] Tuning on the dev sample only (7-point grid, early stopping on 2013Q4 hold-out; the grid
      is flat, 0.350–0.354; chosen: 15 leaves, min 1,000 per leaf, 389 trees)
- [x] Explainability: TreeSHAP vs scorecard contributions (`lcrisk explain`); top decline
      reason agrees 29% (76% within LightGBM's top 3); 15% of LightGBM's top reasons are
      features the scorecard doesn't use
- [x] Cut-off value: at 20% declines on 2015, LightGBM avoids 628 more defaults (~$4.1m)
- [x] Champion/challenger table: Gini, KS, Brier, calibration; is the lift worth the opacity?

## Phase 4: Validation (weeks 3–4, the main point) — `lcrisk validate`, `lcrisk figures`
- [x] Discrimination: Gini with bootstrap CIs, KS, ROC; Gini by issue quarter (stable/rising)
- [x] Calibration: reliability plots, HL, binomial and Jeffreys per master-scale grade (R1–R8).
      Test: scorecard HL p = 0.66. OOT 2015: HL rejects; Jeffreys rejects 7 of 8 grades.
- [x] Stability: score PSI ≤ 0.026 every quarter (green), CSI green except
      verification_status (amber 0.23; LC changed its verification practice; not in the
      scorecard). **Headline: PSI green while default rate drifts up**, so PSI alone
      would not have caught it. Drift is in outcomes given features, not in the features.
- [x] Benchmarks: LC sub-grade (fig 2, fig 6)
- [x] Recalibration: intercept shift fitted on 2014, tested on 2015. Closes ~1/3 of the
      2015 gap (mean PD 12.9% → 13.6% vs 14.9% observed); 2015 kept worsening, so a
      one-year-lagged recalibration is not enough. Recommend quarterly monitoring with
      trigger-based recalibration.
- [x] VIF table (all < 2). CAP curve skipped: ROC + Gini (= accuracy ratio) cover it

## Phase 5: Expected loss and stress (week 5) — `lcrisk el`
- [x] EAD at default = funded − principal received: 50% (R1) to 64% (R8) of funded
- [x] LGD = 1 − net recoveries / EAD: ~0.87–0.90 by grade (unsecured, sold to debt buyers);
      settled loans ~0.71. Downturn LGD = ×1.054 (2007–09 vintages vs dev window)
- [x] 12-month share of lifetime defaults by grade: 24% (R1) to 41% (R8), empirical
- [x] EL backtest: predicted/realised loss 0.93 (2014) → 0.99 recalibrated; 0.81 (2015) →
      0.87 recalibrated. Caveat: 2015 recoveries may be incomplete at the snapshot.
- [x] Stress on the 2015 book ($3.62bn funded): crisis shift +0.19 log-odds (2007–09 at fixed
      sub-grade). Lifetime EL 6.6% baseline → 7.8% downturn PD → 8.2% with downturn LGD.
      Realised 2015 loss was 7.6%: the 2015 vintage came close to a crisis-severity year.
- [x] Vasicek one-year capital (retail ρ, downturn LGD): 10.3% of funded at 99.9%, 6.4% at 99%

## Phase 6: Write-up (week 6)
- [x] Model documentation (`docs/model_doc/lc-pd-model.md` → `lcrisk doc` →
      `reports/model_doc/lc-pd-model.pdf`, 11 pages): document control, purpose/scope, data,
      methodology, performance, validation, EL, stress/capital, monitoring plan, findings
      log (F1–F9), limitations, appendices
- [x] Post for jurstadev.xyz (`jurstadev-portfolio/src/content/posts/credit-risk-model.md`, draft: true until reviewed); PDF copied to `public/writeups/lending-club-pd-model.pdf`
- [x] `lcrisk reproduce` runs everything from the raw CSV (~6 min)

## Review changes (2026-10-27)
External review of the published post and doc raised five points; all addressed:
- [x] Stress anchoring: crisis shift recomputed on four anchors (`el.crisis_shift_range`,
      `reports/tables/stress_anchors.csv`). Sub-grade +0.18 → 2015 stressed EL 8.1%; FICO × DTI
      +0.56 → 11.4%. LC graded crisis-era loans 1–6 notches harsher at fixed FICO, so the
      sub-grade anchor understates. F8 raised to Medium; capital on the FICO × DTI anchor.
      Anchor cells < 50 dev loans dropped (moved the sub-grade shift 0.19 → 0.18).
- [x] "Crisis without a recession" reframed as Lending Club credit-supply / selection drift:
      issuance $2.0bn → $6.4bn (2013 → 2015), full verification 41% → 28%, accounts opened up
      while no-enquiry share rose 52% → 61% (stacking / soft-pull signal), DTI 16.8 → 18.5;
      BLS unemployment 7.4% → 5.3%.
- [x] Miscalibration size: observed ÷ predicted per grade with Jeffreys intervals
      (`metrics.calibration_by_grade`); 2014 rejected grades 1.05–1.10, 2015 1.09–1.21.
      Monitoring trigger now uses the ratio.
- [x] Fairness: disparate impact untestable (no protected attributes, names or full address;
      BISG impossible); stated in §3.3, limitations, new finding F10.
- [x] Reason codes grouped into 8 families (`explain.FAMILIES`): top-reason agreement 29% → 75%;
      93% within LightGBM's top 3.
- [x] `scripts/check_numbers.py` extended to 193 checks; mutation-tested (4 planted errors caught).
