-- RFM (Recency, Frequency, Monetary) feature engineering in SQL
-- Output feeds directly into the ML model as feature columns
-- Shows: multiple CTEs, NTILE(), JULIANDAY(), NULLIF()

WITH snapshot AS (
    SELECT '2018-07-01' AS snapshot_date
),
customer_rfm AS (
    SELECT
        o.customer_id,
        -- Recency: days since last order
        JULIANDAY((SELECT snapshot_date FROM snapshot))
        - JULIANDAY(MAX(o.order_date))                       AS recency_days,
        -- Frequency: number of distinct orders
        COUNT(DISTINCT o.order_id)                           AS frequency,
        -- Monetary: total spend
        ROUND(SUM(oi.price + oi.freight_value), 2)           AS monetary,
        -- Derived
        ROUND(
            SUM(oi.price + oi.freight_value)
            / NULLIF(COUNT(DISTINCT o.order_id), 0),
            2
        )                                                    AS avg_order_value,
        JULIANDAY((SELECT snapshot_date FROM snapshot))
        - JULIANDAY(MIN(o.order_date))                       AS customer_age_days
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'delivered'
      AND o.order_date < (SELECT snapshot_date FROM snapshot)
    GROUP BY o.customer_id
),
rfm_scored AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        avg_order_value,
        customer_age_days,
        -- NTILE splits into 5 buckets; lower recency = higher score
        6 - NTILE(5) OVER (ORDER BY recency_days ASC)        AS r_score,
        NTILE(5)     OVER (ORDER BY frequency ASC)           AS f_score,
        NTILE(5)     OVER (ORDER BY monetary ASC)            AS m_score
    FROM customer_rfm
)
SELECT
    customer_id,
    ROUND(recency_days, 1)                                   AS recency_days,
    frequency,
    monetary,
    avg_order_value,
    ROUND(customer_age_days, 1)                              AS customer_age_days,
    r_score,
    f_score,
    m_score,
    r_score + f_score + m_score                              AS rfm_score,
    CASE
        WHEN r_score + f_score + m_score >= 12 THEN 'Champions'
        WHEN r_score + f_score + m_score >= 9  THEN 'Loyal'
        WHEN r_score >= 4 AND f_score <= 2     THEN 'New Customers'
        WHEN r_score <= 2 AND f_score >= 3     THEN 'At Risk'
        WHEN r_score <= 2                      THEN 'Lost'
        ELSE 'Potential Loyalists'
    END                                                      AS rfm_segment
FROM rfm_scored
ORDER BY rfm_score DESC;
