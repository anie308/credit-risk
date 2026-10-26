"""Recompute every number cited in the model doc (and the blog post, if present) and check
the text still matches. Run after `lcrisk reproduce` and before `lcrisk doc`:

    python scripts/check_numbers.py

Each check formats a value the way the text does and looks for that string; whitespace is
collapsed first, so line wrapping in the Markdown doesn't matter. Claims that are comparisons
rather than numbers ("seven of eight", "never above 0.03") are checked as conditions.
Exits non-zero on any mismatch.
"""

import pickle
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lcrisk import loss  # noqa: E402
from lcrisk.explain import FAMILIES  # noqa: E402
from lcrisk.pipeline import lc_grade_rank  # noqa: E402

T = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"
POST_PATH = (ROOT.parent / "jurstadev-portfolio" / "src" / "content" / "posts"
             / "credit-risk-model.md")
NEW_BUREAU_FIELDS = [
    "tot_cur_bal", "total_rev_hi_lim", "mort_acc", "num_actv_rev_tl", "bc_util",
    "mo_sin_old_rev_tl_op", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "mths_since_recent_inq", "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
]
WORDS = {1: "one", 2: "two", 3: "three", 6: "six", 7: "seven", 8: "eight", 16: "sixteen"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


DOC = _norm((ROOT / "docs" / "model_doc" / "lc-pd-model.md").read_text(encoding="utf-8"))
POST = _norm(POST_PATH.read_text(encoding="utf-8")) if POST_PATH.exists() else None
results: list[tuple[bool, str, str]] = []


def check(where: str, text: str, label: str) -> None:
    """`text` must appear in the doc, the post, or both."""
    targets = {"doc": [DOC], "post": [POST], "both": [DOC, POST]}[where]
    for src in targets:
        if src is None:  # post not present: skip its checks
            continue
        results.append((_norm(text) in src, label, text))


def condition(ok: bool, label: str) -> None:
    results.append((bool(ok), label, "(condition)"))


def f3(x: float) -> str:
    return f"{x:.3f}"


def minus(s: str) -> str:
    return s.replace("-", "−")  # the doc typesets negatives with a true minus sign


pop = pd.read_parquet(PROCESSED / "population.parquet")
scored = pd.read_parquet(PROCESSED / "scored.parquet")
perf = pd.read_csv(PROCESSED / "performance.csv")
wf = pd.read_csv(PROCESSED / "waterfall.csv")
models = pickle.load(open(PROCESSED / "models.pkl", "rb"))
hl = pd.read_csv(T / "hosmer_lemeshow.csv").set_index(["sample", "model"])
cg = pd.read_csv(T / "calibration_grades.csv")
byq = pd.read_csv(T / "by_quarter.csv")
fs = pd.read_csv(T / "feature_stability.csv").set_index("feature")
recal = pd.read_csv(T / "recalibration.csv").set_index(["model", "stage"])
par = pd.read_csv(T / "el_parameters.csv")
bt = pd.read_csv(T / "el_backtest.csv").set_index(["year", "model", "pd"])
st = pd.read_csv(T / "stress.csv").set_index("scenario")
crisis = pd.read_csv(T / "stress_calibration.csv").iloc[0]
vif = pd.read_csv(T / "vif.csv").set_index("feature")["vif"]
iv = pd.read_csv(T / "iv.csv").set_index("feature")
coef = pd.read_csv(T / "scorecard_coefficients.csv").set_index("term")
pts = pd.read_csv(T / "scorecard_points.csv")
imp = pd.read_csv(T / "importance.csv").set_index("feature")
rc = pd.read_csv(T / "reason_codes.csv").iloc[0]
cut = pd.read_csv(T / "cutoff_value.csv").set_index("reject_rate")
tun = pd.read_csv(T / "tuning.csv")


def p(sample: str, model: str, col: str) -> float:
    return perf[(perf["sample"] == sample) & (perf["model"] == model)][col].iloc[0]


# --- data and population
check("doc", f"{int(wf.rows_in.iloc[0]):,}", "raw rows")
check("doc", f"{int(wf.rows_out.iloc[0]):,}", "loans in file")
for _, r in wf.iterrows():
    check("doc", f"| {int(r.rows_in):,} | {int(r.excluded):,} | {int(r.rows_out):,} |",
          f"waterfall: {r.step}")
joint = int(wf.loc[wf.step == "individual application", "excluded"].iloc[0])
check("doc", f"({joint} in the whole", "joint applications")
check("both", f"{len(pop):,}", "population size")
check("doc", f"{pop.bad.mean():.2%}", "population default rate")
dev = pop[pop.issue_d.between("2012-10-01", "2013-12-01")]
check("doc", f"{len(dev):,} loans, {dev.bad.mean():.1%} default rate", "dev window")
for y in (2014, 2015):
    check("doc", f"{(pop.vintage == y).sum():,}", f"OOT {y} size")
n = scored["sample"].value_counts()
check("doc", f"{n['train']:,}", "train size")
check("doc", f"{n['test']:,}", "test size")
q3 = pop[pop.issue_q == "2012Q3"]["tot_cur_bal"].notna().mean()
check("doc", f"{100 * q3:.0f}% populated", "2012Q3 coverage")
absent = [c for c in NEW_BUREAU_FIELDS if pop.loc[pop.vintage <= 2011, c].notna().mean() == 0]
check("both", WORDS.get(len(absent), str(len(absent))), "number of late-added bureau fields")
fico_floor = pop.loc[pop.issue_d >= "2012-10-01", "fico"].min()
condition(fico_floor == 662, "FICO floor 660 (midpoint 662)")
g = pop.pivot_table(index="vintage", columns="grade", values="bad", aggfunc="mean")
for grade in ("C", "E"):
    check("post", f"{100 * g.loc[2013, grade]:.0f}% to {100 * g.loc[2015, grade]:.0f}%",
          f"grade {grade} 2013 -> 2015")

# --- discrimination
for s in ("test", "oot_2014", "oot_2015"):
    for m in ("scorecard", "lightgbm", "lc_subgrade"):
        check("doc", f"{f3(p(s, m, 'gini'))} ({f3(p(s, m, 'gini_lo'))}–{f3(p(s, m, 'gini_hi'))})",
              f"Gini CI {m} {s}")
        check("post", f3(p(s, m, "gini")), f"Gini {m} {s}")
for m in ("scorecard", "lightgbm"):
    check("doc", f"| {f3(p('train', m, 'gini'))} | {f3(p('test', m, 'gini'))} | "
                 f"{f3(p('test', m, 'ks'))} | {p('test', m, 'brier'):.4f} |",
          f"dev performance {m}")
dq, oq = byq[byq.issue_q <= "2013Q4"], byq[byq.issue_q > "2013Q4"]
check("doc", f"({dq.gini_scorecard.min():.3f}–{dq.gini_scorecard.max():.3f})",
      "scorecard dev Gini range")
condition(oq.gini_scorecard.min() >= dq.gini_scorecard.min(), "scorecard OOT quarters >= dev low")
low = oq.loc[oq.gini_lightgbm.idxmin()]
check("doc", f"to {low.gini_lightgbm:.3f} in {low.issue_q}", "LightGBM OOT low")
check("doc", f"development low of {dq.gini_lightgbm.min():.3f}", "LightGBM dev low")
condition((oq.gini_lightgbm < dq.gini_lightgbm.min()).sum() == 1, "LightGBM dips below dev once")

# --- calibration and stability
check("doc", f"p = {hl.loc[('test', 'scorecard'), 'hl_p']:.2f}", "HL p test scorecard")
check("doc", f"| {hl.loc[('test', 'lightgbm'), 'hl_p']:.2f} |", "HL p test LightGBM")
rej = cg[cg.jeffreys_p < 0.01].groupby(["sample", "model"]).size()
condition(rej.get(("test", "scorecard"), 0) + rej.get(("test", "lightgbm"), 0) == 0,
          "no grade rejected on test")
check("doc", f"| {rej.get(('oot_2014', 'scorecard'), 0)} of 8 |", "2014 grades rejected")
seven = WORDS[rej[("oot_2015", "scorecard")]]
check("doc", f"{seven} of eight grades", "2015 grades rejected")
check("post", f"{seven} of the eight grades", "2015 grades rejected")
check("doc", f"{100 * hl.loc[('oot_2015', 'scorecard'), 'mean_pd']:.1f}%", "2015 mean PD scorecard")
check("post", f"{100 * hl.loc[('oot_2015', 'lightgbm'), 'mean_pd']:.1f}% for LightGBM",
      "2015 mean PD LightGBM")
check("doc", f"| {100 * hl.loc[('oot_2014', 'scorecard'), 'mean_pd']:.1f}% | "
             f"{100 * hl.loc[('oot_2014', 'scorecard'), 'odr']:.1f}% |", "2014 PD vs observed")
check("both", f"{100 * hl.loc[('oot_2015', 'scorecard'), 'odr']:.1f}%", "2015 observed rate")
check("doc", f"{byq.psi_score.max():.3f} ({byq.loc[byq.psi_score.idxmax(), 'issue_q']})",
      "max score PSI")
condition(byq.psi_score.max() < 0.03, "score PSI never above 0.03")
check("post", "never went above 0.03", "PSI headline")
condition(list(fs.index[fs.csi_2015 >= 0.1]) == ["verification_status"], "only amber CSI")
v = fs.loc["verification_status"]
check("doc", f"{v.csi_2014:.2f} in 2014, {v.csi_2015:.2f} in 2015", "verification CSI")
check("doc", f"(IV {iv.loc['verification_status', 'iv']:.3f})", "verification IV")
check("doc", f"{100 * imp.loc['verification_status', 'lightgbm']:.1f}% of LightGBM",
      "verification importance")

# --- recalibration
for m, tag in (("scorecard", "Scorecard"), ("lightgbm", "LightGBM")):
    b, a = recal.loc[(m, "before")], recal.loc[(m, "after")]
    check("doc", f"{tag} (+{b['shift']:.3f} log-odds) | {100 * b.mean_pd:.1f}% → "
                 f"{100 * a.mean_pd:.1f}%", f"recalibration {m}")
    check("doc", f"{b.hl_stat:,.0f} → {a.hl_stat:,.0f}", f"recalibration HL {m}")
    check("doc", f"| {int(a.grades_rejected_jeffreys)} of 8 |", f"recalibration rejects {m}")
b, a = recal.loc[("scorecard", "before")], recal.loc[("scorecard", "after")]
closed = (a.mean_pd - b.mean_pd) / (b.odr - b.mean_pd)
condition(0.3 <= closed <= 0.4, f"recalibration closes about a third of the gap ({closed:.2f})")

# --- scorecard
for f in coef.index.drop("const"):
    check("doc", minus(f"| {coef.loc[f, 'coef']:.3f} | {coef.loc[f, 'se']:.3f} | "
                       f"{iv.loc[f, 'iv']:.3f} | {vif[f]:.2f} |"), f"coefficient row {f}")
check("doc", minus(f"| {coef.loc['const', 'coef']:.3f} | {coef.loc['const', 'se']:.3f} |"),
      "intercept")
condition(coef.drop("const")["z"].abs().min() > 4, "all |z| > 4")
condition(vif.max() < 2, "all VIF < 2")
check("doc", f"and {len(pts)} bins", "bin count")
for feat, b_ in [("fico", "(-inf, 662.0]"), ("fico", "(697.0, 707.0]"), ("fico", "(737.0, inf]"),
                 ("acc_open_past_24mths", "(-inf, 0.0]"), ("acc_open_past_24mths", "(7.0, inf]"),
                 ("mths_since_recent_inq", "(-inf, 1.0]"), ("mths_since_recent_inq", "Missing"),
                 ("loan_to_income", "(-inf, 0.109]"), ("loan_to_income", "(0.341, inf]")]:
    rr = pts[(pts.feature == feat) & (pts.bin == b_)].iloc[0]
    check("doc", f"| {int(rr.n):,} | {100 * rr.bad_rate:.1f}% | {int(rr.points)} |",
          f"points {feat} {b_}")
miss = pts[pts.bin == "Missing"].set_index("feature")
check("doc", f"missing, {100 * miss.loc['bc_open_to_buy', 'bad_rate']:.1f}% default rate",
      "no-bankcard default rate")
check("doc", f"missing, {100 * miss.loc['mths_since_recent_inq', 'bad_rate']:.1f}%)",
      "no-enquiry default rate")

# --- challenger
best = tun.iloc[0]
check("doc", f"{int(best.num_leaves)} leaves", "chosen num_leaves")
check("doc", f"{int(best.min_child_samples):,} loans per leaf", "chosen min_child_samples")
check("doc", f"{tun.gini_valid.min():.3f}–{tun.gini_valid.max():.3f}", "tuning range")
check("doc", f"at {models['lightgbm'].booster.best_iteration} trees", "LightGBM trees")
c = cut.loc[0.2]
check("doc", f"| {100 * c.bad_rate_scorecard:.2f}% | {100 * c.bad_rate_lightgbm:.2f}% |",
      "bad rate at 20% declines")
check("doc", f"| {int(c.defaults_avoided_scorecard):,} | {int(c.defaults_avoided_lightgbm):,} "
             f"(+{int(c.extra_defaults_avoided_lgbm)}) |", "defaults avoided")
check("both", f"{int(c.extra_defaults_avoided_lgbm)} more", "extra defaults avoided")
funded15 = bt.loc[(2015, "scorecard", "raw")].funded
per_default = bt.loc[(2015, "scorecard", "raw")].realised_loss / pop[pop.vintage == 2015].bad.sum()
value = int(c.extra_defaults_avoided_lgbm) * per_default
check("both", f"${round(per_default, -2):,.0f}", "loss per default")
check("doc", f"${value / 1e6:.1f} million, {100 * value / funded15:.2f}%", "challenger value")
check("doc", f"{100 * rc.same_top_reason:.0f}% ({100 * rc.scorecard_top_in_lgbm_top3:.0f}% "
             "within its top three)", "reason-code agreement")
check("both", f"{100 * rc.lgbm_top_not_in_scorecard:.0f}%", "top reason outside scorecard")

# --- risk parameters and EL
sp = par[par.model == "scorecard"].set_index("grade_r")
check("doc", f"{100 * sp.loc['R1', 'ead_ratio']:.0f}% of the amount funded in R1 rising to "
             f"{100 * sp.loc['R8', 'ead_ratio']:.0f}% in R8", "EAD ratio range")
check("doc", f"{sp.lgd.min():.2f}–{sp.lgd.max():.2f} by grade", "LGD range")
check("doc", f"{100 * sp.loc['R1', 'share_12m']:.0f}% (R1) to "
             f"{100 * sp.loc['R8', 'share_12m']:.0f}% (R8)", "12-month share range")
rl = loss.realised_loss(dev.merge(scored[["id"]], on="id"))
check("post", f"about **{100 * rl.ead_ratio.mean():.0f}%**", "EAD share (post)")
check("post", f"about **{100 * rl.loss.sum() / rl.ead.sum():.0f}%**", "LGD (post)")
alld = loss.realised_loss(pop).join(pop[["bad", "debt_settlement_flag"]])
alld = alld[alld.bad == 1]
settled = alld[alld.debt_settlement_flag == "Y"]
check("doc", f"settlement ({100 * len(settled) / len(alld):.0f}%) lose less, about "
             f"{settled.loss.sum() / settled.ead.sum():.2f}.", "settled loans")
for y in (2014, 2015):
    for stage in ("raw", "recalibrated"):
        row = bt.loc[(y, "scorecard", stage)]
        check("doc", f"${row.expected_loss / 1e6:.1f}m ({100 * row.el_rate:.2f}%)",
              f"EL {y} {stage}")
        check("doc", f"| {row.ratio:.2f} |", f"EL ratio {y} {stage}")
    row = bt.loc[(y, "scorecard", "raw")]
    check("doc", f"${row.realised_loss / 1e6:.1f}m ({100 * row.loss_rate:.2f}%)", f"realised {y}")

# --- stress and capital
anchors = pd.read_csv(T / "stress_anchors.csv").set_index("anchor")
ANCHOR_NAMES = {"sub_grade": "LC sub-grade", "grade": "LC grade",
                "fico_band": "FICO band (20 points)", "fico_band_x_dti": "FICO band × DTI quintile"}
for a_, r in anchors.iterrows():
    check("doc", f"| {ANCHOR_NAMES[a_]} | {int(r.loans):,} ({100 * r.coverage:.0f}%) | "
                 f"{100 * r.pd_at_dev_rates:.1f}% | {100 * r.odr:.1f}% | +{r['shift']:.2f} | "
                 f"{100 * r.mean_pd_stressed:.1f}% | ${r.lifetime_el / 1e6:.1f}m "
                 f"({100 * r.el_rate_downturn_pd_lgd:.1f}%) |", f"stress anchor {a_}")
condition(abs(crisis["shift"] - anchors.loc["sub_grade", "shift"]) < 1e-12,
          "main stress uses the sub-grade anchor row")
lo_a, hi_a = anchors.loc["sub_grade"], anchors.loc["fico_band_x_dti"]
lo_el, hi_el = 100 * lo_a.el_rate_downturn_pd_lgd, 100 * hi_a.el_rate_downturn_pd_lgd
check("doc", f"+{lo_a['shift']:.2f} log-odds on LC sub-grades", "F8 low shift")
check("doc", f"+{hi_a['shift']:.2f} on FICO × DTI; 2015 stressed EL "
             f"{lo_el:.1f}% to {hi_el:.1f}%",
      "F8 range")
check("doc", f"from 6.6% to {lo_el:.1f}%", "summary low stress")
check("doc", f"to {hi_el:.1f}%. Lending Club graded", "summary high stress")
condition(2.9 <= hi_a["shift"] / lo_a["shift"] <= 3.3, "anchors disagree by a factor of three")
check("doc", f"{100 * (crisis.downturn_lgd_factor - 1):.1f}% higher", "downturn LGD factor")
check("doc", f"about {round(anchors.loans.min(), -3):,.0f} matched loans", "crisis loans matched")
for name, label in (("baseline", "Baseline (recalibrated PD)"),
                    ("downturn PD", "Downturn PD, sub-grade anchor"),
                    ("downturn PD + LGD", "Downturn PD and LGD, sub-grade anchor")):
    row = st.loc[name]
    check("doc", f"| {label} | {100 * row.mean_pd:.1f}% | {row.mean_lgd:.3f} | "
                 f"${row.lifetime_el / 1e6:.1f}m | {100 * row.el_rate:.2f}% |", f"stress {name}")
check("doc", f"| Downturn PD and LGD, FICO × DTI anchor | {100 * hi_a.mean_pd_stressed:.1f}% | | "
             f"${hi_a.lifetime_el / 1e6:.1f}m | {100 * hi_a.el_rate_downturn_pd_lgd:.2f}% |",
      "stress FICO x DTI row")
loss15 = bt.loc[(2015, "scorecard", "raw")].loss_rate
base_rate = st.loc["baseline", "el_rate"]
check("post", f"| Normal conditions | {100 * base_rate:.1f}% |", "post baseline")
check("post", f"| Crisis, matched on Lending Club grade | {lo_el:.1f}% |",
      "post grade stress")
check("post", f"| Crisis, matched on credit score and debt-to-income | "
              f"{hi_el:.1f}% |", "post FICO stress")
check("post", f"| **{100 * loss15:.1f}%** |", "realised 2015 (post)")
pos_lo = (loss15 - base_rate) / (lo_a.el_rate_downturn_pd_lgd - base_rate)
pos_hi = (loss15 - base_rate) / (hi_a.el_rate_downturn_pd_lgd - base_rate)
condition(0.6 <= pos_lo <= 0.7, f"realised about two-thirds to sub-grade stress ({pos_lo:.2f})")
condition(0.18 <= pos_hi <= 0.25, f"realised about a fifth to FICO x DTI stress ({pos_hi:.2f})")

# --- grade harshness by era (sub-grade rank at fixed FICO band)
ranked = pop.assign(rank=lc_grade_rank(pop), band=pop.fico // 20 * 20)
early = ranked[ranked.vintage.between(2007, 2009)].groupby("band")["rank"].mean()
dev_era = ranked[ranked.vintage.between(2012, 2013)].groupby("band")["rank"].mean()
gap = (early - dev_era).loc[660:760]
condition((gap > 0).all(), "crisis-era sub-grades worse in every FICO band")
check("doc", f"rank {early.loc[700]:.1f} against {dev_era.loc[700]:.1f}", "FICO 700 band ranks")
check("post", f"{WORDS[round(gap.min())]} to {WORDS[round(gap.max())]} notches worse",
      "notch range (post)")

# --- calibration size: observed / predicted per grade
sc15 = cg[(cg["sample"] == "oot_2015") & (cg.model == "scorecard")].set_index("grade")
sc14 = cg[(cg["sample"] == "oot_2014") & (cg.model == "scorecard")].set_index("grade")


def _ratio(r):
    dagger = "" if r.jeffreys_p < 0.01 else " †"
    return f"{r.obs_pred:.2f} ({r.obs_pred_lo:.2f}–{r.obs_pred_hi:.2f}){dagger}"


for g_ in sc15.index:
    check("doc", f"| {g_} | {100 * sc15.loc[g_, 'pd_mean']:.1f}% | {_ratio(sc14.loc[g_])} | "
                 f"{_ratio(sc15.loc[g_])} |", f"obs/pred row {g_}")
r15, r14 = sc15[sc15.jeffreys_p < 0.01], sc14[sc14.jeffreys_p < 0.01]
rng15 = f"{100 * (r15.obs_pred.min() - 1):.0f}–{100 * (r15.obs_pred.max() - 1):.0f}%"
rng14 = f"{100 * (r14.obs_pred.min() - 1):.0f}–{100 * (r14.obs_pred.max() - 1):.0f}%"
check("both", rng15, "2015 shortfall range")
check("both", rng14, "2014 shortfall range")
check("doc", f"{r15.obs_pred.min():.2f}–{r15.obs_pred.max():.2f} in the 7 of 8", "F1 ratio range")
check("doc", f"with {round(sc15.n.min(), -3):,.0f}–{round(sc15.n.max(), -3):,.0f} loans per grade",
      "loans per grade 2015")

# --- reason-code families
check("both", f"{100 * rc.same_top_family:.0f}%", "family agreement")
check("doc", f"75% ({100 * rc.scorecard_top_family_in_lgbm_top3:.0f}% within its top three)",
      "family top-3")
outside = 100 * rc.lgbm_top_family_not_in_scorecard
check("doc", f"outside the scorecard's families only {outside:.0f}%",
      "family outside scorecard")
check("post", f"top three **{100 * rc.scorecard_top_family_in_lgbm_top3:.0f}%**",
      "family top-3 (post)")
check("doc", f"into {WORDS[len(FAMILIES)]} reason families",
      "number of families")

# --- platform drift evidence (36-month population unless stated)
vy = pop[pop.vintage.between(2013, 2015)]
ver = pd.crosstab(vy.vintage, vy.verification_status, normalize="index")
check("both", f"{100 * ver.loc[2013, 'Verified']:.0f}%", "full verification 2013")
check("doc", f"{100 * ver.loc[2015, 'Verified']:.0f}% in 2015", "full verification 2015")
check("post", f"to {100 * ver.loc[2015, 'Verified']:.0f}%", "full verification 2015 (post)")
check("doc", f"from {100 * ver.loc[2013, 'Source Verified']:.0f}% to "
             f"{100 * ver.loc[2015, 'Source Verified']:.0f}%", "source verified")
acc = vy.groupby("vintage")["acc_open_past_24mths"].mean()
check("doc", f"{acc[2013]:.1f} to {acc[2015]:.1f} on average", "accounts opened")
noinq = vy.groupby("vintage")["inq_last_6mths"].apply(lambda s: (s == 0).mean())
check("both", f"from {100 * noinq[2013]:.0f}% to {100 * noinq[2015]:.0f}%", "no recent enquiry")
dti_ = vy.groupby("vintage")["dti"].mean()
check("doc", f"from {dti_[2013]:.1f} to {dti_[2015]:.1f}", "mean DTI")
raw = pd.read_csv(ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv",
                  usecols=["id", "issue_d", "funded_amnt"], engine="pyarrow")
raw["y"] = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce").dt.year
iss = raw.groupby("y").agg(n=("id", "size"), amt=("funded_amnt", "sum"))
check("doc", f"${iss.loc[2013, 'amt'] / 1e9:.1f} billion ({int(iss.loc[2013, 'n']):,} loans)",
      "issuance 2013")
check("doc", f"${iss.loc[2014, 'amt'] / 1e9:.1f} billion in 2014", "issuance 2014")
check("doc", f"${iss.loc[2015, 'amt'] / 1e9:.1f} billion ({int(iss.loc[2015, 'n']):,} loans)",
      "issuance 2015")
check("post", f"${iss.loc[2013, 'amt'] / 1e9:.1f} billion in 2013 to "
              f"${iss.loc[2015, 'amt'] / 1e9:.1f} billion in 2015", "issuance (post)")
condition(2.9 <= iss.loc[2015, "amt"] / iss.loc[2013, "amt"] <= 3.5, "tripled in two years")
# External fact, not recomputable here: BLS annual-average unemployment 7.4% (2013),
# 6.2% (2014), 5.3% (2015). Checked by hand against BLS releases on 2026-10-27.
check("both", "7.4% in 2013", "unemployment 2013 cited")
v999, v99 = st.loc["Vasicek 99.9% one-year"], st.loc["Vasicek 99.0% one-year"]
check("doc", f"${v999.capital / 1e6:.1f}m, **{100 * v999.capital_rate:.1f}% of the amount "
             "funded**", "capital at 99.9%")
check("post", f"about **{100 * v999.capital_rate:.0f}%**", "capital (post)")
check("doc", f"${v999.one_year_el / 1e6:.1f}m", "one-year EL")
check("doc", f"At 99% it is {100 * v99.capital_rate:.1f}%", "capital at 99%")
check("doc", f"${funded15 / 1e9:.2f}bn", "2015 funded")

bad = [r for r in results if not r[0]]
print(f"{len(results) - len(bad)} of {len(results)} checks passed")
for _, label, text in bad:
    print(f"MISMATCH {label}: expected {text!r}")
if POST is None:
    print(f"(blog post not found at {POST_PATH}; post checks skipped)")
sys.exit(1 if bad else 0)
