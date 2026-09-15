# SQL rebuild of the loan process analysis

The original project used Python and pm4py. This folder rebuilds the same analysis in SQL (DuckDB) straight from the raw BPI Challenge 2017 event log, so every headline number can be checked without the Python code.

## Run it

```
pip install -r requirements.txt
# put "BPI Challenge 2017.xes.gz" from data.4tu.nl in this folder
python 01_xes_to_csv.py
python 02_build_duckdb.py
python run_sql.py queries/00_build_views.sql output
python run_sql.py queries/03_reproduce_headline_metrics.sql output/03
python run_sql.py queries/01_data_quality_profile.sql output/01
python run_sql.py queries/02_process_analysis.sql output/02
```

Runs in about a minute on a laptop. Results for every query are saved as CSV in `output/`.

## Layers

| Layer | Object | Rule |
|---|---|---|
| Raw | `raw_events` | Exactly as extracted, all text |
| Staging | `events` | Renamed and typed, nothing filtered |
| Clean | `events_clean` | Exact duplicates on case, activity and timestamp removed (0 found) |
| Derived | `cases`, `event_gaps`, `wait_before` | Case metrics, waits between events, variants, final state |

## Headline numbers reproduced in SQL

| Metric | Python project | SQL | Query |
|---|---|---|---|
| Events, applications | 1,202,267 / 31,509 | 1,202,267 / 31,509 | RM01 |
| Cases over a 14 day SLA | 64.1% | 64.1% (20,183 cases) | RM02 |
| Delay before W_Call after offers | 239,868 days, 191,091 occurrences, 30.1 h mean | same | RM04 |
| Repeat events | 712,859 in 100% of cases | same | RM05 |

## What the SQL checks added

| Check | Result | Query |
|---|---|---|
| Repeat events counting completed tasks only | 69,668 repeats in 51.5% of cases. Most of the 712,859 are the same task logged at schedule, start, suspend, resume and complete | RM09 |
| SLA breach by final application state | Paid out (A_Pending) 53.7%, cancelled 86.0%, denied 50.5%. Cancelled applications are 8,966 of the 20,183 breaches | RM10 |
| Where the W_Call after offers delay sits | 207,921 of the 239,868 days (86.7%) are tasks suspended between customer calls | RM11 |
| Time to decision vs time to customer response | Median 0.9 days from creation to first offer, 13.8 days from offer sent to acceptance | PA02, PA09 |

Data: BPI Challenge 2017, van Dongen, B.F., 4TU.ResearchData, doi:10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
