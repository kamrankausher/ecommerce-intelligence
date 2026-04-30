-- A/B experiment data pull
-- Returns conversion summary per variant for statistical testing in Python

SELECT
    experiment_id,
    variant,
    COUNT(*)                                                 AS total_exposed,
    SUM(converted)                                           AS total_converted,
    ROUND(AVG(CAST(converted AS FLOAT)) * 100, 3)            AS conversion_rate_pct,
    MIN(event_date)                                          AS experiment_start,
    MAX(event_date)                                          AS experiment_end
FROM ab_events
GROUP BY experiment_id, variant
ORDER BY experiment_id, variant;

-- Raw events for a specific experiment (swap in experiment_id below)
-- Used by ab_simulator.py to run z-test
SELECT
    variant,
    customer_id,
    converted,
    event_date
FROM ab_events
WHERE experiment_id = 'checkout_button_color'
ORDER BY variant, event_date;
