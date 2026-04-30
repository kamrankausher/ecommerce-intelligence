-- Monthly revenue with running total window function
-- Shows: GROUP BY, SUM() OVER(), ROUND(), date truncation

SELECT
    STRFTIME('%Y-%m', o.order_date)                          AS month,
    ROUND(SUM(oi.price + oi.freight_value), 2)               AS monthly_revenue,
    COUNT(DISTINCT o.order_id)                               AS order_count,
    ROUND(AVG(oi.price + oi.freight_value), 2)               AS avg_order_value,
    ROUND(
        SUM(SUM(oi.price + oi.freight_value))
            OVER (ORDER BY STRFTIME('%Y-%m', o.order_date)
                  ROWS UNBOUNDED PRECEDING),
        2
    )                                                        AS running_total_revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'delivered'
GROUP BY STRFTIME('%Y-%m', o.order_date)
ORDER BY month;
