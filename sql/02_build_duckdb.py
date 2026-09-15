"""Load the BPI Challenge 2017 event log CSV into DuckDB and create the staging view.

Run order:
    python 01_xes_to_csv.py          # BPI Challenge 2017.xes.gz -> event_log_raw.csv
    python 02_build_duckdb.py        # event_log_raw.csv -> bpi2017.duckdb (raw_events, events)
    python run_sql.py queries/00_build_views.sql output
    python run_sql.py queries/03_reproduce_headline_metrics.sql output/03
    python run_sql.py queries/01_data_quality_profile.sql output/01
    python run_sql.py queries/02_process_analysis.sql output/02
"""
import duckdb

con = duckdb.connect("bpi2017.duckdb")
con.execute("SET TimeZone='UTC'")

# Raw layer: exactly as extracted, every column kept as text.
con.execute("""
CREATE OR REPLACE TABLE raw_events AS
SELECT * FROM read_csv('event_log_raw.csv', header=true, all_varchar=true)
""")

# Staging layer: business-readable names and types, nothing filtered.
con.execute("""
CREATE OR REPLACE VIEW events AS
SELECT "case:concept:name"      AS application_id,
       "concept:name"           AS activity,
       "org:resource"           AS actor,
       "EventOrigin"            AS event_origin,
       "EventID"                AS event_id,
       "lifecycle:transition"   AS lifecycle,
       "Action"                 AS action,
       CAST(CAST("time:timestamp" AS TIMESTAMPTZ) AT TIME ZONE 'UTC' AS TIMESTAMP) AS event_ts,
       "case:LoanGoal"          AS loan_purpose,
       "case:ApplicationType"   AS application_type,
       TRY_CAST("case:RequestedAmount" AS DOUBLE) AS requested_amount,
       TRY_CAST("OfferedAmount" AS DOUBLE)        AS offered_amount,
       TRY_CAST("CreditScore" AS DOUBLE)          AS credit_score,
       TRY_CAST("MonthlyCost" AS DOUBLE)          AS monthly_cost,
       TRY_CAST("NumberOfTerms" AS DOUBLE)        AS term_months,
       "Accepted"               AS offer_accepted,
       "Selected"               AS offer_selected,
       "OfferID"                AS offer_id
FROM raw_events
""")
print(con.execute("SELECT COUNT(*) AS events, COUNT(DISTINCT application_id) AS applications FROM events").fetchall())
