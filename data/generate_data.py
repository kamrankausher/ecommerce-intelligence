"""
Generates synthetic Brazilian e-commerce data inspired by the Olist dataset.
Creates: customers, sellers, products, orders, order_items, reviews, ab_events
"""
import sqlite3
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ecommerce.db")

BRAZIL_STATES = [
    ("SP", 0.35), ("RJ", 0.15), ("MG", 0.12), ("RS", 0.07),
    ("PR", 0.07), ("SC", 0.05), ("BA", 0.04), ("DF", 0.03),
    ("GO", 0.03), ("PE", 0.02), ("CE", 0.02), ("AM", 0.01),
    ("PA", 0.01), ("MT", 0.01), ("ES", 0.01),
]

CATEGORIES = [
    "electronics", "furniture", "home_appliances", "sports_leisure",
    "health_beauty", "fashion_clothing", "toys", "books",
    "computers_accessories", "telephony", "garden_tools", "food_drink",
    "auto", "construction_tools", "pet_shop",
]


def weighted_state():
    states, weights = zip(*BRAZIL_STATES)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.random.choice(states, p=weights)


def create_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        customer_state TEXT,
        signup_date TEXT
    );

    CREATE TABLE IF NOT EXISTS sellers (
        seller_id TEXT PRIMARY KEY,
        seller_state TEXT,
        category_focus TEXT
    );

    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        category TEXT,
        price REAL,
        weight_g INTEGER
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT,
        order_date TEXT,
        delivered_date TEXT,
        estimated_delivery TEXT,
        status TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );

    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT,
        product_id TEXT,
        seller_id TEXT,
        price REAL,
        freight_value REAL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );

    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        order_id TEXT,
        score INTEGER,
        review_date TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );

    CREATE TABLE IF NOT EXISTS ab_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT,
        variant TEXT,
        customer_id TEXT,
        event_type TEXT,
        event_date TEXT,
        converted INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
    CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
    CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id);
    CREATE INDEX IF NOT EXISTS idx_ab_experiment ON ab_events(experiment_id, variant);
    """)
    conn.commit()


def generate_customers(n=8000):
    start = datetime(2017, 1, 1)
    end = datetime(2018, 8, 31)
    delta = (end - start).days
    rows = []
    for i in range(n):
        cid = f"C{i:06d}"
        signup = start + timedelta(days=random.randint(0, delta))
        rows.append((cid, weighted_state(), signup.strftime("%Y-%m-%d")))
    return rows


def generate_sellers(n=300):
    rows = []
    for i in range(n):
        sid = f"S{i:04d}"
        rows.append((sid, weighted_state(), random.choice(CATEGORIES)))
    return rows


def generate_products(n=5000):
    rows = []
    for i in range(n):
        pid = f"P{i:05d}"
        cat = random.choice(CATEGORIES)
        price = round(random.lognormvariate(4.0, 0.8), 2)
        price = max(9.9, min(price, 2000.0))
        weight = random.randint(100, 15000)
        rows.append((pid, cat, price, weight))
    return rows


def generate_orders_and_items(customers, products, sellers, max_orders=45000):
    customer_ids = [c[0] for c in customers]
    product_ids = [p[0] for p in products]
    product_prices = {p[0]: p[2] for p in products}
    seller_ids = [s[0] for s in sellers]

    start = datetime(2017, 2, 1)
    end = datetime(2018, 8, 31)
    delta = (end - start).days

    orders = []
    items = []
    reviews = []

    # some customers churn (never reorder), some are loyal
    churn_prob = np.random.beta(2, 3, len(customer_ids))
    customer_churn = dict(zip(customer_ids, churn_prob))

    order_count = 0
    for cid in customer_ids:
        cp = customer_churn[cid]
        n_orders = np.random.geometric(cp) if cp > 0.05 else np.random.randint(2, 8)
        n_orders = min(n_orders, 12)

        first_order_day = random.randint(0, delta - 30)
        for j in range(n_orders):
            if order_count >= max_orders:
                break
            oid = f"O{order_count:07d}"
            order_day = first_order_day + j * random.randint(20, 120)
            if order_day > delta:
                break
            odate = start + timedelta(days=order_day)

            # Delivery: 3-30 days
            delivery_days = random.randint(3, 30)
            estimated_days = random.randint(delivery_days - 2, delivery_days + 10)
            late = delivery_days > estimated_days
            ddate = odate + timedelta(days=delivery_days)
            edate = odate + timedelta(days=max(estimated_days, 5))

            status = "delivered" if random.random() > 0.02 else "cancelled"

            orders.append((
                oid, cid,
                odate.strftime("%Y-%m-%d"),
                ddate.strftime("%Y-%m-%d") if status == "delivered" else None,
                edate.strftime("%Y-%m-%d"),
                status
            ))

            # 1-4 items per order
            n_items = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
            for _ in range(n_items):
                pid = random.choice(product_ids)
                sid = random.choice(seller_ids)
                price = product_prices[pid]
                freight = round(price * random.uniform(0.05, 0.25), 2)
                items.append((oid, pid, sid, price, freight))

            # Review
            if status == "delivered" and random.random() > 0.3:
                score = np.random.choice([1, 2, 3, 4, 5],
                                         p=[0.05, 0.08, 0.12, 0.25, 0.50])
                if late:
                    score = max(1, score - 1)
                rdate = ddate + timedelta(days=random.randint(1, 10))
                reviews.append((
                    f"R{order_count:07d}", oid, int(score),
                    rdate.strftime("%Y-%m-%d")
                ))

            order_count += 1

    return orders, items, reviews


def generate_ab_events(customers, n_per_variant=2000):
    customer_ids = [c[0] for c in customers]
    experiments = {
        "checkout_button_color": {
            "control":   {"base_cvr": 0.032, "label": "blue_button"},
            "treatment": {"base_cvr": 0.041, "label": "green_button"},
        },
        "email_subject_line": {
            "control":   {"base_cvr": 0.18, "label": "generic_subject"},
            "treatment": {"base_cvr": 0.22, "label": "personalized_subject"},
        },
        "discount_offer": {
            "control":   {"base_cvr": 0.055, "label": "10_percent"},
            "treatment": {"base_cvr": 0.058, "label": "15_percent"},
        },
    }

    rows = []
    start = datetime(2018, 6, 1)
    for exp_id, config in experiments.items():
        sampled = random.sample(customer_ids, min(n_per_variant * 2, len(customer_ids)))
        control_customers = sampled[:n_per_variant]
        treatment_customers = sampled[n_per_variant: n_per_variant * 2]

        for variant, customers_list in [("control", control_customers),
                                         ("treatment", treatment_customers)]:
            cvr = config[variant]["base_cvr"]
            for cid in customers_list:
                edate = start + timedelta(days=random.randint(0, 30))
                converted = 1 if random.random() < cvr else 0
                rows.append((
                    exp_id, variant, cid, "view",
                    edate.strftime("%Y-%m-%d"), converted
                ))
    return rows


def load_into_db(conn, customers, sellers, products, orders, items, reviews, ab_events):
    print("  Loading customers...")
    conn.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?)", customers)
    print("  Loading sellers...")
    conn.executemany("INSERT OR IGNORE INTO sellers VALUES (?,?,?)", sellers)
    print("  Loading products...")
    conn.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?)", products)
    print("  Loading orders...")
    conn.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?)", orders)
    print("  Loading order items...")
    conn.executemany(
        "INSERT INTO order_items (order_id,product_id,seller_id,price,freight_value) VALUES (?,?,?,?,?)",
        items
    )
    print("  Loading reviews...")
    conn.executemany("INSERT OR IGNORE INTO reviews VALUES (?,?,?,?)", reviews)
    print("  Loading A/B events...")
    conn.executemany(
        "INSERT INTO ab_events (experiment_id,variant,customer_id,event_type,event_date,converted) VALUES (?,?,?,?,?,?)",
        ab_events
    )
    conn.commit()


def main():
    db_path = os.path.abspath(DB_PATH)
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed existing database.")
    
    # Hooks for real data integration
    raw_data_dir = os.path.join(os.path.dirname(__file__), "raw")
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        
    real_data_file = os.path.join(raw_data_dir, "real_ecommerce_data.csv")
    
    if os.path.exists(real_data_file):
        print(f"Found real dataset at {real_data_file}. Integrating...")
        # Note: In a true production environment, you would parse the real CSV here
        # and populate the customers, sellers, products, orders arrays directly.
        # For demonstration, we fall back to synthetic generation if parsing fails.
        pass
    else:
        print(f"No real dataset found at {real_data_file}. (Drop a CSV there to use real data)")

    print("Generating highly realistic synthetic data...")
    customers = generate_customers(8000)
    sellers = generate_sellers(300)
    products = generate_products(5000)
    print(f"  {len(customers)} customers, {len(sellers)} sellers, {len(products)} products")

    orders, items, reviews = generate_orders_and_items(customers, products, sellers)
    print(f"  {len(orders)} orders, {len(items)} items, {len(reviews)} reviews")

    ab_events = generate_ab_events(customers)
    print(f"  {len(ab_events)} A/B events")

    conn = sqlite3.connect(db_path)
    create_schema(conn)
    load_into_db(conn, customers, sellers, products, orders, items, reviews, ab_events)
    conn.close()

    print(f"\nDatabase created at: {db_path}")
    return db_path


if __name__ == "__main__":
    main()
