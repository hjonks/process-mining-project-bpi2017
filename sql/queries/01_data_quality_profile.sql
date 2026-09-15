/* =========================================================================
   PROJECT KESTREL — DATA QUALITY PROFILE
   Source: ORIGIN-4 origination extract, Jan 2016 – Feb 2017
   Author: [YOUR NAME]        Date: [DATE]        Version: 0.1 DRAFT
   =========================================================================

   PURPOSE
   Establish whether this extract can support the analysis before any analysis
   is done. A BA who reports a finding from data they have not profiled will
   eventually report a finding that is an artefact of the extract, and will be
   found out in front of the people who know the system.

   HOW TO USE THIS FILE
   Every query is followed by a FINDING line. Fill each one in, in your own
   words, with a number in it. Then write the IMPLICATION -- what it means for
   the requirements, not just what it means for the data. The queries are the
   scaffold; the findings are the deliverable, and the findings are yours.

   Run with:  python run_query.py 01_data_quality_profile.sql
   ========================================================================= */


-- @@ DQ01 — Extract scope: volumes and date range
-- Does the extract match what you were told you were given? Check it. Extracts
-- are routinely mis-specified and it is your name on the number.
SELECT
    COUNT(*)                              AS total_events,
    COUNT(DISTINCT application_id)        AS total_applications,
    COUNT(DISTINCT activity)              AS distinct_activities,
    COUNT(DISTINCT actor)                 AS distinct_actors,
    MIN(event_ts)                         AS earliest_event,
    MAX(event_ts)                         AS latest_event,
    DATE_DIFF('day', MIN(event_ts), MAX(event_ts)) AS span_days
FROM events;
-- FINDING:
-- IMPLICATION:


-- @@ DQ02 — Completeness: NULL rate by column
-- A high NULL rate is not automatically a defect. Offer fields are NULL on
-- events that are not offer events, which is correct. Your job is to separate
-- structurally-absent from actually-missing, and to say which is which.
SELECT 'application_id'   AS column_name, COUNT(*) FILTER (WHERE application_id  IS NULL) AS nulls, COUNT(*) AS rows FROM events
UNION ALL SELECT 'activity',          COUNT(*) FILTER (WHERE activity         IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'actor',             COUNT(*) FILTER (WHERE actor            IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'event_ts',          COUNT(*) FILTER (WHERE event_ts         IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'lifecycle',         COUNT(*) FILTER (WHERE lifecycle        IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'loan_purpose',      COUNT(*) FILTER (WHERE loan_purpose     IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'requested_amount',  COUNT(*) FILTER (WHERE requested_amount IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'credit_score',      COUNT(*) FILTER (WHERE credit_score     IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'offered_amount',    COUNT(*) FILTER (WHERE offered_amount   IS NULL), COUNT(*) FROM events
UNION ALL SELECT 'offer_id',          COUNT(*) FILTER (WHERE offer_id         IS NULL), COUNT(*) FROM events
ORDER BY nulls DESC;
-- FINDING:
-- WHICH NULLS ARE STRUCTURAL, WHICH ARE DEFECTS:
-- IMPLICATION:


-- @@ DQ03 — Uniqueness: duplicate event IDs
-- Deepak mentioned a manual retry pattern. Tom mentioned a save-and-reopen
-- workaround. Both would show up here or in DQ04. Does the evidence support them?
SELECT COUNT(*) AS duplicated_event_ids
FROM (SELECT event_id FROM events GROUP BY event_id HAVING COUNT(*) > 1);
-- FINDING:
-- IMPLICATION:


-- @@ DQ04 — Duplicate activity events: same case, same activity, same second
-- A different and more interesting duplicate: not a repeated ID, but the same
-- thing apparently happening twice at once. Usually a system artefact. Decide
-- whether to exclude these from timing analysis and DOCUMENT the decision.
SELECT COUNT(*) AS suspect_duplicate_pairs
FROM (
    SELECT application_id, activity, DATE_TRUNC('second', event_ts) AS ts_sec, COUNT(*) AS n
    FROM events GROUP BY 1,2,3 HAVING COUNT(*) > 1
);
-- FINDING:
-- YOUR TREATMENT DECISION (exclude / retain / flag) AND WHY:


-- @@ DQ05 — Referential integrity: events vs case table
-- Two files were extracted separately. Do they agree? If not, which is right?
SELECT
    (SELECT COUNT(DISTINCT application_id) FROM events)                              AS ids_in_event_log,
    (SELECT COUNT(DISTINCT application_id) FROM cases)                               AS ids_in_case_table,
    (SELECT COUNT(*) FROM (SELECT DISTINCT application_id FROM events
        EXCEPT SELECT DISTINCT application_id FROM cases))                           AS in_events_not_cases,
    (SELECT COUNT(*) FROM (SELECT DISTINCT application_id FROM cases
        EXCEPT SELECT DISTINCT application_id FROM events))                          AS in_cases_not_events;
-- FINDING:
-- IMPLICATION FOR WHICH SOURCE YOU TRUST:


-- @@ DQ06 — Temporal integrity: events out of sequence
-- Timestamps that go backwards inside a case break every duration calculation
-- downstream. Find them before they find you.
SELECT COUNT(*) AS out_of_order_events
FROM event_gaps
WHERE wait_hours < 0;
-- FINDING:
-- IMPLICATION:


-- @@ DQ07 — Boundary effects: truncated cases
-- Cases open at the start of the window have no beginning; cases open at the end
-- have no end. Both distort average duration. How many are affected, and what
-- will you do about them? This is the single most common analytical error in
-- process data and stating that you controlled for it is a strong signal.
WITH bounds AS (SELECT MIN(event_ts) AS lo, MAX(event_ts) AS hi FROM events)
SELECT
    COUNT(*) FILTER (WHERE c.case_start <= b.lo + INTERVAL 7 DAY)  AS starts_in_first_week,
    COUNT(*) FILTER (WHERE c.case_end   >= b.hi - INTERVAL 7 DAY)  AS ends_in_last_week,
    COUNT(*)                                                        AS all_cases
FROM cases c CROSS JOIN bounds b;
-- FINDING:
-- YOUR TREATMENT DECISION AND WHY:


-- @@ DQ08 — Value validity: requested amount distribution
-- Look for impossible values, defaults masquerading as data, and clustering at
-- round numbers that indicates manual entry rather than customer input.
SELECT
    MIN(requested_amount) AS min_amt, MAX(requested_amount) AS max_amt,
    AVG(requested_amount) AS mean_amt,
    MEDIAN(requested_amount) AS median_amt,
    COUNT(*) FILTER (WHERE requested_amount IS NULL)  AS null_amt,
    COUNT(*) FILTER (WHERE requested_amount <= 0)     AS non_positive_amt,
    COUNT(*) FILTER (WHERE requested_amount % 1000 = 0) AS exact_thousands
FROM cases;
-- FINDING:
-- IMPLICATION:


-- @@ DQ09 — Categorical integrity: outcome values
-- Are the outcome categories what the business thinks they are? Compare this
-- against what Tom and Rachel believe the outcome split looks like. If the
-- extract cannot tell you how many applications were approved, say so loudly.
SELECT outcome, COUNT(*) AS cases,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM cases GROUP BY 1 ORDER BY cases DESC;
-- FINDING:
-- IS THIS FIT FOR PURPOSE FOR THE COMMITTEE'S QUESTIONS? WHY / WHY NOT:


-- @@ DQ10 — Actor integrity: system vs human actors
-- Some actors are people, some are batch processes. Timing analysis that treats
-- a nightly batch as a person will mislead you.
SELECT actor, COUNT(*) AS events, COUNT(DISTINCT application_id) AS cases,
       MIN(event_ts) AS first_seen, MAX(event_ts) AS last_seen
FROM events GROUP BY 1 ORDER BY events DESC LIMIT 25;
-- FINDING (which actors look non-human, and how can you tell):
-- IMPLICATION:


-- @@ DQ11 — Granularity check: lifecycle transitions and event origin
-- Deepak said some activities are logged at state-change level rather than user
-- action. This is where you verify that. It determines what the log can and
-- cannot tell you about human effort.
SELECT event_origin, lifecycle, action, COUNT(*) AS events
FROM events GROUP BY 1,2,3 ORDER BY events DESC;
-- FINDING:
-- WHAT THE LOG CANNOT TELL YOU AS A RESULT:


-- @@ DQ12 — The measurement question: what start point does the data support?
-- The Charter says "14 working days from receiving everything we need".
-- The Ops MI measures calendar days from application creation. Martin flagged
-- the discrepancy. Can this extract even identify the point at which a file
-- became complete? Answer that here, honestly, because it determines whether
-- the organisation can measure its own published promise.
SELECT activity, COUNT(*) AS occurrences, COUNT(DISTINCT application_id) AS cases
FROM events
WHERE lower(activity) LIKE '%complete%' OR lower(activity) LIKE '%submit%'
   OR lower(activity) LIKE '%accept%'   OR lower(activity) LIKE '%valid%'
GROUP BY 1 ORDER BY occurrences DESC;
-- FINDING:
-- CAN MERIDIAN MEASURE ITS OWN CHARTER COMMITMENT? YES / NO / PARTIALLY, AND WHY:
-- THIS IS THE MOST IMPORTANT FINDING IN THE PROFILE. WRITE IT PROPERLY.


-- @@ DQ13 — Overall fitness statement
-- Not a query. Write three sentences here and lift them into the assessment:
--   1. What this extract is fit for.
--   2. What it is not fit for, specifically.
--   3. What additional data you would request, from whom, and why it matters.
SELECT 'Write your fitness statement in the comment block above' AS reminder;
