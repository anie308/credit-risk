from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "reports" / "figures"

# Population: 36-month loans that could all run to maturity before the snapshot
TERM_MONTHS = 36
LAST_ISSUE = "2015-12-01"

# Development window: from 2012Q4, when every bureau field is populated, to end-2013.
# Later years are the out-of-time sample. 2007-2012Q3 is used only to calibrate the stress.
DEV_START, DEV_END = "2012-10-01", "2013-12-01"
OOT_YEARS = (2014, 2015)
SEED = 42
