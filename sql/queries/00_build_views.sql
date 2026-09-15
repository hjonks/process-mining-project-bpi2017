/* Build layer: every case-level metric is derived here in SQL from the raw
   BPI Challenge 2017 event log, so no Python output is needed downstream. */

-- Clean events: same rule as the Python pipeline (drop exact duplicates on
-- case + activity + timestamp), kept as a view so the exclusion is auditable.
CREATE OR REPLACE VIEW events_clean AS
SELECT * EXCLUDE (rn) FROM (
    SELECT e.*, ROW_NUMBER() OVER (PARTITION BY application_id, activity, event_ts ORDER BY event_id) AS rn
    FROM events e
) WHERE rn = 1;

-- Wait from each event to the next, attributed to the activity the case is
-- sitting AT (the Kestrel harness definition, used by PA03 and PA04).
CREATE OR REPLACE VIEW event_gaps AS
SELECT
    application_id, activity, actor, event_ts,
    LEAD(activity) OVER w                                        AS next_activity,
    LEAD(event_ts) OVER w                                        AS next_ts,
    DATE_DIFF('millisecond', event_ts, LEAD(event_ts) OVER w) / 3600000.0 AS wait_hours,
    ROW_NUMBER() OVER w                                          AS event_seq
FROM events_clean
WINDOW w AS (PARTITION BY application_id ORDER BY event_ts, event_id);

-- Wait before each event, attributed to the activity being waited FOR
-- (the definition used in the original Python bottleneck analysis).
CREATE OR REPLACE VIEW wait_before AS
SELECT
    application_id, activity, lifecycle, event_ts,
    LAG(activity)  OVER wb AS prev_activity,
    LAG(lifecycle) OVER wb AS prev_lifecycle,
    DATE_DIFF('millisecond', LAG(event_ts) OVER wb, event_ts) / 3600000.0 AS wait_hours
FROM events_clean
WINDOW wb AS (PARTITION BY application_id ORDER BY event_ts, event_id);

-- One row per application.
CREATE OR REPLACE VIEW cases AS
WITH base AS (
    SELECT application_id,
           MIN(event_ts) AS case_start, MAX(event_ts) AS case_end,
           COUNT(*) AS n_events, COUNT(DISTINCT activity) AS n_unique_activities,
           COUNT(DISTINCT actor) AS n_resources,
           ARG_MAX(activity, event_ts) AS last_activity,
           ANY_VALUE(loan_purpose) AS loan_purpose, ANY_VALUE(application_type) AS application_type,
           ANY_VALUE(requested_amount) AS requested_amount
    FROM events_clean GROUP BY application_id
),
variants AS (
    SELECT application_id, STRING_AGG(activity, ' > ' ORDER BY event_ts) AS variant
    FROM events_clean GROUP BY application_id
),
variant_rank AS (
    SELECT variant, DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) AS variant_rank
    FROM variants GROUP BY variant
)
SELECT b.*,
    DATE_DIFF('millisecond', case_start, case_end) / 86400000.0 AS duration_days,
    14 AS sla_days,
    CASE WHEN DATE_DIFF('millisecond', case_start, case_end) / 86400000.0 > 14 THEN 1 ELSE 0 END AS sla_breach,
    CASE
        WHEN last_activity IN ('A_Accepted','A_Complete') THEN 'Approved'
        WHEN last_activity = 'A_Denied' THEN 'Denied'
        WHEN last_activity = 'O_Cancelled' THEN 'Cancelled'
        WHEN last_activity LIKE 'W_%' THEN 'Still Processing'
        WHEN last_activity IN ('O_Sent (mail and online)','O_Sent (online only)','O_Returned') THEN 'Offer Sent'
        ELSE 'Unknown'
    END AS outcome,
    e.end_state,
    vr.variant_rank,
    vr.variant_rank <= 3  AS is_top3_variant,
    vr.variant_rank <= 10 AS is_top10_variant
FROM base b
JOIN variants v USING (application_id)
JOIN variant_rank vr USING (variant)
LEFT JOIN (
    -- Final application state from the A_ events (A_Pending = offer accepted and loan paid out)
    SELECT application_id, ARG_MAX(activity, event_ts) AS end_state
    FROM events_clean WHERE activity IN ('A_Pending','A_Denied','A_Cancelled')
    GROUP BY application_id
) e USING (application_id);
