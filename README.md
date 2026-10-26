# Lending Club PD Model

*A WoE/logistic scorecard, a LightGBM challenger, out-of-time validation, and expected loss under stress.*

A retail probability-of-default model built and validated the way a bank's model risk team would
review it, on Lending Club's public loan book (36-month loans, 2007–2015 vintages).

- Model documentation, in the format of a validation report:
  [reports/model_doc/lc-pd-model.pdf](reports/model_doc/lc-pd-model.pdf)
- Plain-English write-up:
  [I built a bank-style credit model. The health check said it was fine while losses rose](https://jurstadev.xyz/articles/credit-risk-model/)

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
```

Download the data (1.7 GB; source and checksum in `data/SOURCES.md`):

```bash
mkdir -p data/raw
curl -L -o data/raw/accepted_2007_to_2018Q4.csv   https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv
sha256sum data/raw/accepted_2007_to_2018Q4.csv   # 3eae03c2...0933d1f36a
```

Then:

```bash
lcrisk reproduce   # everything below, ~6 minutes
lcrisk ingest      # population filters + target -> data/processed/population.parquet
lcrisk profile     # coverage and default rates by vintage
lcrisk fit         # scorecard + LightGBM on 2012Q4-2013, scored on test and OOT 2014/2015
lcrisk validate    # calibration tests, PSI/CSI, recalibration -> reports/tables/
lcrisk el          # LGD/EAD, EL backtest, stress, Vasicek capital
lcrisk explain     # VIF, SHAP vs scorecard, reason codes, cut-off value
lcrisk figures     # reports/figures/
lcrisk doc         # model documentation PDF (needs Chrome/Edge; pip install -e ".[report]")
```

After any refit, run `python scripts/check_numbers.py` before `lcrisk doc`. It recomputes every
number cited in the model documentation (and the blog post, if the portfolio repo sits next to
this one) and fails if the text no longer matches.

Use Python 3.12 or later, and avoid LightGBM 4.7.0: its Windows wheel crashes on any label.

## Results

| Gini | Test | Out of time 2014 | Out of time 2015 |
|---|---|---|---|
| WoE scorecard (champion) | 0.310 | 0.316 | 0.330 |
| LightGBM (challenger) | 0.352 | 0.354 | 0.363 |
| Lending Club sub-grade | 0.277 | 0.319 | 0.357 |

Ranking holds out of time, but calibration does not: 2015 defaults ran at 14.9% against a mean
PD of 12.9%, while the score PSI stayed below 0.03. Full write-up:
[reports/model_doc/lc-pd-model.pdf](reports/model_doc/lc-pd-model.pdf).

## Layout

```
src/lcrisk/
  ingest.py     raw CSV -> modelling population, exclusion waterfall      (phase 1)
  profile.py    feature coverage and default rates by vintage             (phase 1)
  woe.py        monotonic WoE binning, information value                 (phase 2)
  scorecard.py  logistic regression on WoE, points scaling               (phase 2)
  metrics.py    Gini/KS, PSI, Hosmer-Lemeshow, binomial/Jeffreys tests   (phase 4)
  samples.py    dev window train/test, out-of-time years                  (phase 2)
  pipeline.py   fit both models, score every sample                       (phase 2-3)
  challenger.py LightGBM with monotone constraints, SHAP                  (phase 3)
  validate.py   master scale, calibration, stability, recalibration      (phase 4)
  loss.py       EAD, LGD, EL, log-odds stress, Vasicek capital            (phase 5)
  el.py         EL backtest and stress scenarios                          (phase 5)
  explain.py    VIF, importance, reason codes, cut-off value              (phase 3-4)
  figures.py    documentation figures
  report/       Markdown -> HTML -> PDF builder
docs/model_doc/ model documentation source
reports/        tables, figures, model documentation PDF
```

See [docs/PLAN.md](docs/PLAN.md) for the phase checklist.
