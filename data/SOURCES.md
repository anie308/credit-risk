# Data sources

## Lending Club accepted loans, 2007–2018Q4

- File: `data/raw/accepted_2007_to_2018Q4.csv` (not in git)
- Original: Kaggle dataset `wordsforthewise/lending-club`, a compilation of Lending Club's
  public loan-level downloads.
- Fetched 2026-10-26 from the CC0 mirror
  `https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv`
- Size 1,675,133,810 bytes; sha256
  `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`
- 151 columns. The last rows are Lending Club's "Total amount funded in policy code N"
  summary lines, not loans; `ingest` drops them with the loan-id filter.
