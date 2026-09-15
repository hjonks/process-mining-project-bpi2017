/* Reproduces the four headline numbers from the Python and pm4py project in
   plain SQL, so each one can be checked independently of the original code. */

-- @@ RM01 Volumes after cleaning
SELECT COUNT(*) AS events_clean, COUNT(DISTINCT application_id) AS applications
FROM events_clean;

-- @@ RM02 SLA breach rate against a 14 calendar day SLA (all cases)
SELECT COUNT(*) AS cases, SUM(sla_breach) AS breaches,
       ROUND(100.0 * AVG(sla_breach), 1) AS breach_rate_pct
FROM cases;

-- @@ RM03 SLA breach rate by outcome
SELECT outcome, COUNT(*) AS cases, ROUND(100.0 * AVG(sla_breach), 1) AS breach_rate_pct
FROM cases GROUP BY outcome ORDER BY cases DESC;

-- @@ RM04 Bottleneck ranking: waiting time before each activity
-- Same filter as the Python code: keep waits from 0 hours up to 60 days.
SELECT activity,
       COUNT(*)                          AS occurrences,
       ROUND(AVG(wait_hours), 1)         AS mean_wait_hours,
       ROUND(MEDIAN(wait_hours), 1)      AS median_wait_hours,
       ROUND(SUM(wait_hours) / 24, 0)    AS total_wait_days,
       ROUND(AVG(wait_hours) * COUNT(*), 0) AS bottleneck_score
FROM wait_before
WHERE wait_hours >= 0 AND wait_hours < 24 * 60
GROUP BY activity
ORDER BY bottleneck_score DESC
LIMIT 10;

-- @@ RM05 Rework: repeat visits to the same activity within a case
WITH per_act AS (
    SELECT application_id, activity, COUNT(*) AS n
    FROM events_clean GROUP BY 1, 2
)
SELECT SUM(n - 1) FILTER (WHERE n > 1)                           AS rework_events,
       COUNT(DISTINCT application_id) FILTER (WHERE n > 1)       AS cases_with_rework,
       ROUND(100.0 * COUNT(DISTINCT application_id) FILTER (WHERE n > 1)
             / COUNT(DISTINCT application_id), 1)                AS pct_cases_with_rework
FROM per_act;

-- @@ RM06 Most reworked activities
WITH per_act AS (
    SELECT application_id, activity, COUNT(*) AS n
    FROM events_clean GROUP BY 1, 2
)
SELECT activity, COUNT(*) AS cases_with_repeat, SUM(n - 1) AS extra_events
FROM per_act WHERE n > 1
GROUP BY activity ORDER BY extra_events DESC LIMIT 10;

-- @@ RM07 SLA breach rate by application start month
SELECT STRFTIME(case_start, '%Y-%m') AS start_month, COUNT(*) AS cases,
       ROUND(100.0 * AVG(sla_breach), 1) AS breach_rate_pct
FROM cases GROUP BY 1 ORDER BY 1;

-- @@ RM08 Case duration percentiles (days)
SELECT ROUND(QUANTILE_CONT(duration_days, 0.25), 1) AS p25,
       ROUND(MEDIAN(duration_days), 1) AS median,
       ROUND(QUANTILE_CONT(duration_days, 0.75), 1) AS p75,
       ROUND(QUANTILE_CONT(duration_days, 0.90), 1) AS p90
FROM cases;

-- @@ RM09 Check 1: how much of the rework count is lifecycle logging?
-- W_ work items log schedule, start, suspend, resume and complete as separate
-- events, so a single task can appear five times. Count completed repeats only.
WITH per_act AS (
    SELECT application_id, activity, COUNT(*) AS n
    FROM events_clean WHERE lifecycle = 'complete' GROUP BY 1, 2
)
SELECT SUM(n - 1) FILTER (WHERE n > 1)                           AS repeat_completions,
       COUNT(DISTINCT application_id) FILTER (WHERE n > 1)       AS cases_with_repeat,
       ROUND(100.0 * COUNT(DISTINCT application_id) FILTER (WHERE n > 1)
             / COUNT(DISTINCT application_id), 1)                AS pct_cases
FROM per_act;

-- @@ RM10 Check 2: SLA breach rate by final application state
SELECT COALESCE(end_state, 'No end state in window') AS end_state, COUNT(*) AS cases,
       ROUND(100.0 * AVG(sla_breach), 1) AS breach_rate_pct,
       ROUND(MEDIAN(duration_days), 1) AS median_days
FROM cases GROUP BY 1 ORDER BY cases DESC;

-- @@ RM11 Check 3: what kind of waiting sits behind the W_Call after offers delay
SELECT prev_lifecycle || ' to ' || lifecycle AS transition,
       prev_activity = activity              AS within_same_task,
       COUNT(*)                              AS occurrences,
       ROUND(AVG(wait_hours), 1)             AS mean_wait_hours,
       ROUND(SUM(wait_hours) / 24, 0)        AS total_wait_days
FROM wait_before
WHERE activity = 'W_Call after offers' AND wait_hours >= 0 AND wait_hours < 24 * 60
GROUP BY 1, 2 ORDER BY total_wait_days DESC LIMIT 8;
