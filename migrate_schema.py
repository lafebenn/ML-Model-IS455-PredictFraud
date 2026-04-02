"""
Migrate orders.is_fraud to nullable INTEGER DEFAULT NULL.
NULL = not scored yet; 0 = not fraud; 1 = fraud.

All existing rows get is_fraud = NULL (per assignment). Other columns unchanged.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("SHOP_DB_PATH", Path(__file__).resolve().parent / "shop.db"))


def main() -> None:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH.resolve()}")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(orders)")
        cols = [row[1] for row in cur.fetchall()]
        if not cols:
            raise RuntimeError("Table 'orders' not found.")

        if "is_fraud" not in cols:
            raise RuntimeError("Column is_fraud not found on orders.")

        cur.execute("SELECT COUNT(*) FROM orders")
        (row_count,) = cur.fetchone()

        # Same layout as original shop.db; only is_fraud line changes (nullable, DEFAULT NULL).
        cur.execute("PRAGMA foreign_keys = OFF")
        cur.execute("BEGIN IMMEDIATE")

        cur.execute(
            """
            CREATE TABLE orders_new (
              order_id           INTEGER PRIMARY KEY AUTOINCREMENT,
              customer_id        INTEGER NOT NULL,
              order_datetime     TEXT NOT NULL,
              billing_zip        TEXT,
              shipping_zip       TEXT,
              shipping_state     TEXT,
              payment_method     TEXT NOT NULL,
              device_type        TEXT NOT NULL,
              ip_country         TEXT NOT NULL,
              promo_used         INTEGER NOT NULL DEFAULT 0,
              promo_code         TEXT,
              order_subtotal     REAL NOT NULL,
              shipping_fee       REAL NOT NULL,
              tax_amount         REAL NOT NULL,
              order_total        REAL NOT NULL,
              risk_score         REAL NOT NULL,
              is_fraud           INTEGER DEFAULT NULL,
              FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
            """
        )

        col_list = ", ".join(cols)
        select_parts = [c if c != "is_fraud" else "NULL" for c in cols]
        select_sql = ", ".join(select_parts)
        cur.execute(f"INSERT INTO orders_new ({col_list}) SELECT {select_sql} FROM orders")

        # One new row per existing order; all get is_fraud = NULL
        updated = row_count
        cur.execute("DROP TABLE orders")
        cur.execute("ALTER TABLE orders_new RENAME TO orders")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_datetime ON orders(order_datetime)")

        # Keep AUTOINCREMENT aligned with max(order_id) after explicit-ID copy
        cur.execute("SELECT MAX(order_id) FROM orders")
        (max_id,) = cur.fetchone()
        if max_id is not None:
            cur.execute("SELECT 1 FROM sqlite_sequence WHERE name = 'orders'")
            if cur.fetchone():
                cur.execute(
                    "UPDATE sqlite_sequence SET seq = ? WHERE name = 'orders'",
                    (max_id,),
                )
            else:
                cur.execute(
                    "INSERT INTO sqlite_sequence (name, seq) VALUES ('orders', ?)",
                    (max_id,),
                )

        conn.commit()
        print(
            f"Migration complete: {DB_PATH.resolve()}\n"
            f"  Recreated orders with is_fraud INTEGER DEFAULT NULL (nullable).\n"
            f"  Rows copied with is_fraud set to NULL: {updated} (table had {row_count} rows)."
        )

        cur.execute("PRAGMA table_info(orders)")
        print("\nNew orders table schema (PRAGMA table_info):")
        for cid, name, ctype, notnull, dflt_value, pk in cur.fetchall():
            nn = "NOT NULL" if notnull else "NULL ok"
            dv = "" if dflt_value is None else f" DEFAULT {dflt_value}"
            pkf = " PRIMARY KEY" if pk else ""
            print(f"  {name}: {ctype} {nn}{dv}{pkf}")

    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error:
            pass
        conn.close()


if __name__ == "__main__":
    main()
