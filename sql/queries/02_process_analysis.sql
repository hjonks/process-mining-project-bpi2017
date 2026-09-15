/* =========================================================================
   PROJECT KESTREL — PROCESS ANALYSIS
   Author: [YOUR NAME]   Date: [DATE]   Version: 0.1 DRAFT
   PREREQUISITE: complete 01_data_quality_profile.sql first. Do not analyse
   data you have not profiled.
   =========================================================================

   These queries compute. They do not conclude. Every one is followed by a
   FINDING line and, where it matters, a CHALLENGE line asking what would have
   to be true for the obvious reading to be wrong. Fill both in.

   Run with:  python run_query.py 02_process_analysis.sql
   ========================================================================= */


-- @@ PA01 — How long does it actually take? Distribution, not average.
-- Martin will reject a mean. Give him the shape.
SELECT
    COUNT(*)                                          AS cases,
    ROUND(AVG(duration_days), 1)                      AS mean_days,
    ROUND(MEDIAN(duration_days), 1)                   AS median_days,
    ROUND(QUANTILE_CONT(duration_days, 0.75), 1)      AS p75_days,
    ROUND(QUANTILE_CONT(duration_days, 0.90), 1)      AS p90_days,
    ROUND(QUANTILE_CONT(duration_days, 0.95), 1)      AS p95_days,
    ROUND(MAX(duration_days), 1)                      AS max_days
FROM cases;
-- FINDING:
-- WHY THE MEAN AND MEDIAN DIFFER, AND WHICH ONE THE BOARD SHOULD SEE:


-- @@ PA02 — The reconciliation: why do three internal reports disagree?
-- Ops says 12 days. Contact Centre says 9. Complaints sampling says 24.
-- Compute turnaround three defensible ways and see which assumption produces
-- which number. This single query answers the Committee's first question and
-- it is the most quietly impressive thing you can put in the pack.
WITH milestones AS (
    SELECT
        application_id,
        MIN(event_ts)                                                          AS created_ts,
        MIN(event_ts) FILTER (WHERE activity ILIKE '%Complete application%')   AS file_complete_ts,
        MIN(event_ts) FILTER (WHERE activity ILIKE 'O_Create Offer%')          AS offer_ts,
        MAX(event_ts)                                                          AS last_ts
    FROM events GROUP BY 1
)
SELECT
    'A. Creation to last event (Ops MI basis)'      AS measure_basis,
    ROUND(AVG(DATE_DIFF('hour', created_ts, last_ts)/24.0), 1)       AS mean_days,
    ROUND(MEDIAN(DATE_DIFF('hour', created_ts, last_ts)/24.0), 1)    AS median_days
FROM milestones
UNION ALL SELECT
    'B. Creation to first offer (decision issued)',
    ROUND(AVG(DATE_DIFF('hour', created_ts, offer_ts)/24.0), 1),
    ROUND(MEDIAN(DATE_DIFF('hour', created_ts, offer_ts)/24.0), 1)
FROM milestones WHERE offer_ts IS NOT NULL
UNION ALL SELECT
    'C. File complete to first offer (Charter wording)',
    ROUND(AVG(DATE_DIFF('hour', file_complete_ts, offer_ts)/24.0), 1),
    ROUND(MEDIAN(DATE_DIFF('hour', file_complete_ts, offer_ts)/24.0), 1)
FROM milestones WHERE offer_ts IS NOT NULL AND file_complete_ts IS NOT NULL;
-- FINDING — which basis produces which internal number:
-- WHICH BASIS DOES THE CUSTOMER EXPERIENCE? (this is the one that matters)
-- YOUR RECOMMENDED SINGLE DEFINITION, AND HOW YOU WILL GET IT AGREED:


-- @@ PA03 — Where does the waiting actually happen?
-- Wait attributed to the activity the case is sitting AT. Note both the mean
-- (how bad is each instance) and the total (how much time in aggregate). They
-- rank differently and the difference is the whole analysis.
SELECT
    activity,
    COUNT(*)                                        AS occurrences,
    ROUND(AVG(wait_hours), 2)                       AS mean_wait_hrs,
    ROUND(MEDIAN(wait_hours), 3)                    AS median_wait_hrs,
    ROUND(QUANTILE_CONT(wait_hours, 0.90), 1)       AS p90_wait_hrs,
    ROUND(SUM(wait_hours)/24.0, 0)                  AS total_wait_days
FROM event_gaps
WHERE wait_hours IS NOT NULL AND wait_hours >= 0
GROUP BY 1
ORDER BY total_wait_days DESC;
-- FINDING — highest TOTAL wait:
-- FINDING — highest MEAN wait:
-- CHALLENGE: an activity with a huge total but a near-zero median is not a slow
--            activity. What is it actually telling you? Answer before you go on.


-- @@ PA04 — The attribution question, tested the other way
-- Same waits, attributed to what the case is waiting FOR rather than where it
-- is sitting. If the ranking changes, your bottleneck claim depends on an
-- analytical choice -- and you must say which you used and why.
SELECT
    next_activity,
    COUNT(*)                          AS occurrences,
    ROUND(AVG(wait_hours), 2)         AS mean_wait_hrs,
    ROUND(SUM(wait_hours)/24.0, 0)    AS total_wait_days
FROM event_gaps
WHERE wait_hours IS NOT NULL AND wait_hours >= 0 AND next_activity IS NOT NULL
GROUP BY 1 ORDER BY total_wait_days DESC LIMIT 15;
-- DOES THE RANKING CHANGE? WHICH ATTRIBUTION DID YOU ADOPT AND WHY:


-- @@ PA05 — Is it a volume problem? (Tom's hypothesis, tested properly)
-- Monthly volume against monthly performance. If delay tracks volume, Tom is
-- right and the answer is capacity. If it does not, the answer is elsewhere.
-- Test his claim honestly. Do not set it up to fail.
SELECT
    DATE_TRUNC('month', case_start)                 AS month,
    COUNT(*)                                        AS applications,
    ROUND(AVG(duration_days), 1)                    AS mean_days,
    ROUND(MEDIAN(duration_days), 1)                 AS median_days,
    ROUND(100.0 * AVG(sla_breach), 1)               AS pct_breaching
FROM cases GROUP BY 1 ORDER BY 1;
-- FINDING — does breach rate track volume?
-- IF NOT VOLUME, WHAT ELSE MOVES WITH IT (look at the same table again):
-- HOW WILL YOU TELL TOM? (write the actual sentence you would say to him)


-- @@ PA06 — Rework: how often does work come back?
-- Tom said roughly one in five. Test it, and separate rework that is inherent
-- to the process from rework that indicates a defect.
SELECT
    activity,
    COUNT(*)                                                     AS total_events,
    COUNT(DISTINCT application_id)                               AS cases_touched,
    ROUND(COUNT(*)::DOUBLE / COUNT(DISTINCT application_id), 2)  AS avg_repeats_per_case,
    MAX(per_case)                                                AS max_repeats_one_case
FROM (
    SELECT application_id, activity, COUNT(*) OVER (PARTITION BY application_id, activity) AS per_case
    FROM events
) GROUP BY 1
HAVING COUNT(*)::DOUBLE / COUNT(DISTINCT application_id) > 1.2
ORDER BY avg_repeats_per_case DESC;
-- FINDING:
-- WHICH REPEATS ARE DESIGNED AND WHICH ARE FAILURE DEMAND:


-- @@ PA07 — Handoffs: how many times does a case change hands?
-- Every handoff is a queue. This is where process cost hides.
WITH h AS (
    SELECT application_id, actor,
           LAG(actor) OVER (PARTITION BY application_id ORDER BY event_ts) AS prev_actor
    FROM events
)
SELECT
    COUNT(*) FILTER (WHERE actor <> prev_actor)::DOUBLE
      / COUNT(DISTINCT application_id)            AS mean_handoffs_per_case,
    COUNT(DISTINCT application_id)                AS cases
FROM h WHERE prev_actor IS NOT NULL;
-- FINDING:
-- IMPLICATION FOR THE TO-BE DESIGN:


-- @@ PA08 — Timing effects: when do things happen, and when do they stall?
-- Sian described a 06:00 batch and a Friday problem. Deepak confirmed nightly
-- integration. Does the log show it? If it does, you have found a cause that is
-- cheap to fix, which is exactly what Rachel asked for.
SELECT
    DAYNAME(event_ts)                          AS day_of_week,
    EXTRACT(hour FROM event_ts)                AS hour_of_day,
    COUNT(*)                                   AS events,
    ROUND(AVG(wait_hours), 2)                  AS mean_wait_after
FROM event_gaps
WHERE wait_hours IS NOT NULL AND wait_hours >= 0
GROUP BY 1,2 ORDER BY 1,2;
-- FINDING — is there a day-of-week effect?
-- FINDING — is there an hour-of-day effect?
-- CHEAPEST INTERVENTION THIS SUGGESTS:


-- @@ PA09 — The offer response loop (Sian's territory)
-- How long between an offer going out and the customer responding, and how much
-- chasing happens in between? This is the largest single block of elapsed time
-- in most origination processes and it is nobody's KPI.
WITH offers AS (
    SELECT application_id,
           MIN(event_ts) FILTER (WHERE activity ILIKE 'O_Sent%')     AS sent_ts,
           MIN(event_ts) FILTER (WHERE activity ILIKE 'O_Accepted%') AS accepted_ts,
           COUNT(*) FILTER (WHERE activity ILIKE 'W_Call after offers%') AS chase_events
    FROM events GROUP BY 1
)
SELECT
    COUNT(*)                                                           AS offers_sent,
    COUNT(accepted_ts)                                                 AS offers_accepted,
    ROUND(100.0 * COUNT(accepted_ts) / COUNT(*), 1)                    AS pct_accepted,
    ROUND(AVG(DATE_DIFF('hour', sent_ts, accepted_ts)/24.0), 1)        AS mean_days_to_accept,
    ROUND(MEDIAN(DATE_DIFF('hour', sent_ts, accepted_ts)/24.0), 1)     AS median_days_to_accept,
    ROUND(AVG(chase_events), 1)                                        AS mean_chase_events,
    MAX(chase_events)                                                  AS max_chase_events
FROM offers WHERE sent_ts IS NOT NULL;
-- FINDING:
-- WHAT PROPORTION OF TOTAL ELAPSED TIME SITS IN THIS LOOP:
-- WHOSE PROBLEM IS THIS CURRENTLY, AND WHOSE SHOULD IT BE:


-- @@ PA10 — Does chasing work? (test before you recommend changing it)
-- If more calls produced more acceptances you would leave it alone. Do they?
WITH offers AS (
    SELECT application_id,
           COUNT(*) FILTER (WHERE activity ILIKE 'W_Call after offers%') AS chases,
           MAX(CASE WHEN activity ILIKE 'O_Accepted%' THEN 1 ELSE 0 END) AS accepted
    FROM events GROUP BY 1
)
SELECT
    LEAST(chases, 10)                       AS chase_attempts,
    COUNT(*)                                AS cases,
    ROUND(100.0 * AVG(accepted), 1)         AS pct_accepted
FROM offers GROUP BY 1 ORDER BY 1;
-- FINDING — where does the marginal return on an additional call disappear?
-- THE REQUIREMENT THIS IMPLIES:
-- CHALLENGE: this is correlational. Cases that get chased more are different
--            from cases that get chased less. Say so in the pack, and say what
--            you would need to test it properly. Priya WILL ask.


-- @@ PA11 — Segment fairness (Martin's Consumer Duty question)
-- Is delay concentrated on any group? Amount band is the best proxy available
-- in this extract -- note that limitation explicitly, because the vulnerability
-- flag Martin mentioned is NOT in the extract and that absence is itself a finding.
SELECT
    CASE WHEN requested_amount < 5000  THEN 'A. under 5k'
         WHEN requested_amount < 10000 THEN 'B. 5k-10k'
         WHEN requested_amount < 15000 THEN 'C. 10k-15k'
         WHEN requested_amount < 20000 THEN 'D. 15k-20k'
         ELSE 'E. 20k plus' END                    AS amount_band,
    COUNT(*)                                        AS cases,
    ROUND(AVG(duration_days), 1)                    AS mean_days,
    ROUND(QUANTILE_CONT(duration_days, 0.90), 1)    AS p90_days,
    ROUND(100.0 * AVG(sla_breach), 1)               AS pct_breaching
FROM cases WHERE requested_amount IS NOT NULL
GROUP BY 1 ORDER BY 1;
-- FINDING:
-- WHAT YOU CANNOT ANSWER WITH THIS EXTRACT, AND WHAT YOU WOULD REQUEST:


-- @@ PA12 — Complexity vs duration: what actually predicts a slow case?
-- Your model said unique activities, resources and rework dominate. Confirm it
-- descriptively here so the finding stands on its own without the model.
SELECT
    n_unique_activities,
    COUNT(*)                            AS cases,
    ROUND(AVG(duration_days), 1)        AS mean_days,
    ROUND(100.0 * AVG(sla_breach), 1)   AS pct_breaching
FROM cases GROUP BY 1 HAVING COUNT(*) > 50 ORDER BY 1;
-- FINDING:
-- THE ONE-SENTENCE VERSION FOR THE BOARD:


-- @@ PA13 — Variant concentration: is there a standard process at all?
SELECT
    is_top3_variant, is_top10_variant,
    COUNT(*)                            AS cases,
    ROUND(AVG(duration_days), 1)        AS mean_days,
    ROUND(100.0 * AVG(sla_breach), 1)   AS pct_breaching
FROM cases GROUP BY 1,2 ORDER BY cases DESC;
-- FINDING:
-- DOES PROCESS STANDARDISATION LOOK LIKE A LEVER? WHAT IS THE EVIDENCE:


-- @@ PA14 — Benefit baseline: size the prize before you design the fix
-- Do not design a solution before you know what the problem is worth. This is
-- your business case denominator and every benefit claim traces back to it.
SELECT
    COUNT(*)                                                          AS breaching_cases,
    ROUND(SUM(duration_days - sla_days), 0)                           AS total_excess_days,
    ROUND(AVG(duration_days - sla_days), 1)                           AS mean_excess_days_per_breach,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM cases), 1)         AS pct_of_all_cases
FROM cases WHERE sla_breach = 1;
-- FINDING:
-- CONVERT THIS TO MONEY. State every assumption you use, in the pack, with a
-- source. An assumption you have labelled is evidence of judgement; an
-- unlabelled one is a hole in your credibility.
