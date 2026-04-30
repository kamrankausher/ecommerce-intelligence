-- Seller performance with RANK() window function
-- Shows: PARTITION BY, RANK(), CASE WHEN inside aggregation, LEFT JOIN

WITH seller_metrics AS (
    SELECT
        oi.seller_id,
        s.seller_state,
        s.category_focus,
        COUNT(DISTINCT oi.order_id)                          AS total_orders,
        ROUND(SUM(oi.price), 2)                              AS total_revenue,
        ROUND(AVG(oi.price), 2)                              AS avg_item_price,
        ROUND(AVG(r.score), 2)                               AS avg_review_score,
        COUNT(r.review_id)                                   AS review_count,
        ROUND(
            AVG(CASE
                WHEN o.delivered_date > o.estimated_delivery
                THEN 1.0 ELSE 0.0
            END) * 100,
            1
        )                                                    AS late_delivery_pct
    FROM order_items oi
    JOIN sellers s        ON oi.seller_id = s.seller_id
    JOIN orders o         ON oi.order_id  = o.order_id
    LEFT JOIN reviews r   ON o.order_id   = r.order_id
    WHERE o.status = 'delivered'
    GROUP BY oi.seller_id
    HAVING total_orders >= 5
)
SELECT
    seller_id,
    seller_state,
    category_focus,
    total_orders,
    total_revenue,
    avg_item_price,
    avg_review_score,
    late_delivery_pct,
    RANK() OVER (
        PARTITION BY seller_state
        ORDER BY total_revenue DESC
    )                                                        AS rank_in_state,
    RANK() OVER (
        ORDER BY total_revenue DESC
    )                                                        AS overall_rank
FROM seller_metrics
ORDER BY overall_rank
LIMIT 50;
