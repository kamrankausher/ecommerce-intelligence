-- Monthly cohort retention analysis
-- Shows: CTEs, DATE_TRUNC equivalent, self-join, division for rate

WITH first_orders AS (
    -- Each customer's first purchase month (their cohort)
    SELECT
        customer_id,
        STRFTIME('%Y-%m', MIN(order_date)) AS cohort_month
    FROM orders
    WHERE status = 'delivered'
    GROUP BY customer_id
),
monthly_activity AS (
    -- All months each customer placed an order
    SELECT
        o.customer_id,
        STRFTIME('%Y-%m', o.order_date) AS active_month
    FROM orders o
    WHERE status = 'delivered'
    GROUP BY o.customer_id, STRFTIME('%Y-%m', o.order_date)
),
cohort_matrix AS (
    -- Join cohort to activity, compute period offset in months
    SELECT
        fo.cohort_month,
        ma.active_month,
        CAST(
            (JULIANDAY(ma.active_month || '-01')
           - JULIANDAY(fo.cohort_month || '-01')) / 30.0
        AS INTEGER)                                  AS period,
        COUNT(DISTINCT ma.customer_id)               AS retained_users
    FROM first_orders fo
    JOIN monthly_activity ma ON fo.customer_id = ma.customer_id
    GROUP BY fo.cohort_month, ma.active_month
),
cohort_sizes AS (
    -- Size of each cohort (period 0 = acquisition month)
    SELECT cohort_month, retained_users AS cohort_size
    FROM cohort_matrix
    WHERE period = 0
)
SELECT
    cm.cohort_month,
    cm.period,
    cm.retained_users,
    cs.cohort_size,
    ROUND(
        CAST(cm.retained_users AS FLOAT) / cs.cohort_size * 100,
        1
    )                                                AS retention_pct
FROM cohort_matrix cm
JOIN cohort_sizes cs ON cm.cohort_month = cs.cohort_month
WHERE cm.cohort_month >= '2017-03'
  AND cm.period <= 6
ORDER BY cm.cohort_month, cm.period;
