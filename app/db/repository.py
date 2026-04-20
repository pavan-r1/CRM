from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.db.schema import get_connection


def to_df(query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    conn = get_connection()
    with conn:
        conn.execute(query, params)
    conn.close()


def execute_many(query: str, rows: list[tuple[Any, ...]]) -> None:
    conn = get_connection()
    with conn:
        conn.executemany(query, rows)
    conn.close()


def count_customers() -> int:
    df = to_df("SELECT COUNT(*) AS c FROM customers")
    return int(df.iloc[0]["c"])


def list_customers(region: str | None = None, plan_tier: str | None = None) -> pd.DataFrame:
    where = []
    params: list[Any] = []
    if region:
        where.append("region = ?")
        params.append(region)
    if plan_tier:
        where.append("plan_tier = ?")
        params.append(plan_tier)
    sql = "SELECT * FROM customers"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY company_name"
    return to_df(sql, tuple(params))


def get_customer(customer_id: int) -> pd.DataFrame:
    return to_df("SELECT * FROM customers WHERE id = ?", (customer_id,))


def upsert_customer(record: dict[str, Any]) -> None:
    fields = (
        record["company_name"],
        record["region"],
        record["plan_tier"],
        record["contract_start"],
        record["contract_end"],
        int(record["nps_score"]),
        record["email"],
        int(record.get("churn_label", 0)),
    )
    if record.get("id"):
        execute(
            """
            UPDATE customers
            SET company_name=?, region=?, plan_tier=?, contract_start=?, contract_end=?,
                nps_score=?, email=?, churn_label=?
            WHERE id=?
            """,
            fields + (int(record["id"]),),
        )
    else:
        execute(
            """
            INSERT INTO customers(company_name, region, plan_tier, contract_start, contract_end,
                                  nps_score, email, churn_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fields,
        )


def delete_customer(customer_id: int) -> None:
    execute("DELETE FROM customers WHERE id = ?", (customer_id,))


def list_tickets(customer_id: int | None = None) -> pd.DataFrame:
    if customer_id is None:
        return to_df(
            """
            SELECT t.*, c.company_name
            FROM tickets t
            JOIN customers c ON c.id = t.customer_id
            ORDER BY t.created_at DESC
            """
        )
    return to_df(
        "SELECT * FROM tickets WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    )


def upsert_ticket(record: dict[str, Any]) -> None:
    fields = (
        int(record["customer_id"]),
        record["created_at"],
        record["severity"],
        record["status"],
        record["subject"],
        int(record["resolution_days"]),
    )
    if record.get("id"):
        execute(
            """
            UPDATE tickets
            SET customer_id=?, created_at=?, severity=?, status=?, subject=?, resolution_days=?
            WHERE id=?
            """,
            fields + (int(record["id"]),),
        )
    else:
        execute(
            """
            INSERT INTO tickets(customer_id, created_at, severity, status, subject, resolution_days)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            fields,
        )


def delete_ticket(ticket_id: int) -> None:
    execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))


def list_devices(customer_id: int | None = None) -> pd.DataFrame:
    if customer_id is None:
        return to_df(
            """
            SELECT d.*, c.company_name
            FROM devices d
            JOIN customers c ON c.id = d.customer_id
            ORDER BY d.install_date DESC
            """
        )
    return to_df(
        "SELECT * FROM devices WHERE customer_id = ? ORDER BY install_date DESC",
        (customer_id,),
    )


def upsert_device(record: dict[str, Any]) -> None:
    fields = (
        int(record["customer_id"]),
        record["device_type"],
        record["model"],
        record["status"],
        record["install_date"],
    )
    if record.get("id"):
        execute(
            """
            UPDATE devices
            SET customer_id=?, device_type=?, model=?, status=?, install_date=?
            WHERE id=?
            """,
            fields + (int(record["id"]),),
        )
    else:
        execute(
            """
            INSERT INTO devices(customer_id, device_type, model, status, install_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            fields,
        )


def delete_device(device_id: int) -> None:
    execute("DELETE FROM devices WHERE id = ?", (device_id,))


def replace_table(table_name: str, df: pd.DataFrame) -> None:
    conn = get_connection()
    with conn:
        conn.execute(f"DELETE FROM {table_name}")
        df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()


def save_health_score(customer_id: int, score: float, details: dict[str, Any]) -> None:
    execute(
        """
        INSERT INTO health_scores(customer_id, calculated_at, score, details_json)
        VALUES (?, ?, ?, ?)
        """,
        (customer_id, datetime.now(timezone.utc).isoformat(), float(score), json.dumps(details)),
    )


def latest_health_scores() -> pd.DataFrame:
    return to_df(
        """
        SELECT hs.customer_id, hs.score, hs.calculated_at, hs.details_json
        FROM health_scores hs
        JOIN (
            SELECT customer_id, MAX(id) AS latest_id
            FROM health_scores
            GROUP BY customer_id
        ) q ON hs.id = q.latest_id
        ORDER BY hs.score ASC
        """
    )


def save_churn_prediction(customer_id: int, probability: float, churn_flag: int, top_factors: list[str]) -> None:
    execute(
        """
        INSERT INTO churn_predictions(customer_id, predicted_at, churn_probability, churn_flag, top_factors_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            customer_id,
            datetime.now(timezone.utc).isoformat(),
            float(probability),
            int(churn_flag),
            json.dumps(top_factors),
        ),
    )


def latest_churn_predictions() -> pd.DataFrame:
    return to_df(
        """
        SELECT cp.customer_id, cp.churn_probability, cp.churn_flag, cp.top_factors_json, cp.predicted_at
        FROM churn_predictions cp
        JOIN (
            SELECT customer_id, MAX(id) AS latest_id
            FROM churn_predictions
            GROUP BY customer_id
        ) q ON cp.id = q.latest_id
        ORDER BY cp.churn_probability DESC
        """
    )


def save_nl_context(session_id: str, context: dict[str, Any]) -> None:
    execute(
        """
        INSERT INTO nl_sessions(session_id, context_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET context_json=excluded.context_json, updated_at=excluded.updated_at
        """,
        (session_id, json.dumps(context), datetime.now(timezone.utc).isoformat()),
    )


def get_nl_context(session_id: str) -> dict[str, Any]:
    df = to_df("SELECT context_json FROM nl_sessions WHERE session_id = ?", (session_id,))
    if df.empty:
        return {}
    return json.loads(df.iloc[0]["context_json"])


def save_generated_email(customer_id: int, subject: str, body: str) -> None:
    execute(
        """
        INSERT INTO generated_emails(customer_id, generated_at, subject, body)
        VALUES (?, ?, ?, ?)
        """,
        (customer_id, datetime.now(timezone.utc).isoformat(), subject, body),
    )
