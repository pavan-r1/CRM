from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from app.ai import llm_adapter
from app.db import repository


REGION_MAP = {
    "na": "North America",
    "north america": "North America",
    "emea": "EMEA",
    "apac": "APAC",
    "latam": "LATAM",
}

PLAN_MAP = {
    "starter": "Starter",
    "professional": "Professional",
    "enterprise": "Enterprise",
}


@dataclass
class QueryResult:
    title: str
    summary: str
    dataframe: pd.DataFrame
    context: dict


def get_or_create_session_id(session_state: dict) -> str:
    if "nl_session_id" not in session_state:
        session_state["nl_session_id"] = str(uuid.uuid4())
    return str(session_state["nl_session_id"])


def _extract_region(query: str) -> str | None:
    q = query.lower()
    for key, value in REGION_MAP.items():
        if key in q:
            return value
    return None


def _extract_plan(query: str) -> str | None:
    q = query.lower()
    for key, value in PLAN_MAP.items():
        if key in q:
            return value
    return None


def _build_summary(title: str, df: pd.DataFrame, context: dict) -> str:
    region_text = context.get("region") or "all regions"
    plan_text = context.get("plan_tier") or "all plan tiers"

    if df.empty:
        return f"No records found for {title.lower()} in {region_text} and {plan_text}."

    if title == "Churn risk":
        high = int((df.get("churn_probability", pd.Series(dtype=float)) >= 0.6).sum())
        total = len(df)
        enterprise = int((df.get("plan_tier", pd.Series(dtype=object)) == "Enterprise").sum())
        return (
            f"Found {total} churn records in {region_text}; {high} are high-risk and "
            f"{enterprise} are enterprise clients."
        )

    if title == "Customer list":
        total = len(df)
        enterprise = int((df.get("plan_tier", pd.Series(dtype=object)) == "Enterprise").sum())
        return f"Found {total} customers in {region_text}; {enterprise} are enterprise clients."

    if title == "Open tickets":
        total = len(df)
        critical = int((df.get("severity", pd.Series(dtype=object)) == "Critical").sum())
        return f"Found {total} open tickets in {region_text}, including {critical} critical issues."

    return f"Found {len(df)} records for {title.lower()} filtered by {region_text} and {plan_text}."


def _schema_hint() -> str:
    return (
        "customers(id, company_name, region, plan_tier, contract_start, contract_end, nps_score, email, churn_label)\n"
        "tickets(id, customer_id, created_at, severity, status, subject, resolution_days)\n"
        "monthly_usage(id, customer_id, month, gb_used, active_devices, incidents)\n"
        "health_scores(id, customer_id, calculated_at, score, details_json)\n"
        "churn_predictions(id, customer_id, predicted_at, churn_probability, churn_flag, top_factors_json)"
    )


def _is_safe_sql(sql_text: str) -> bool:
    cleaned = sql_text.strip().lower()
    if not cleaned.startswith("select"):
        return False
    blocked = ["insert", "update", "delete", "drop", "alter", "create", "pragma", ";"]
    if any(token in cleaned for token in blocked):
        return False
    return True


def _run_llm_query(user_query: str, context: dict, region: str | None, plan: str | None) -> tuple[str, pd.DataFrame] | None:
    context_hint = (
        f"region={context.get('region')}, plan_tier={context.get('plan_tier')}, "
        f"last_intent={context.get('last_intent')}"
    )
    sql_text = llm_adapter.sql_from_nl(user_query, _schema_hint(), context_hint=context_hint)
    if not sql_text:
        return None

    sql_clean = sql_text.strip().replace("```sql", "").replace("```", "").strip()
    if not _is_safe_sql(sql_clean):
        return None

    try:
        df = repository.to_df(sql_clean)
    except Exception:
        return None

    if region and "region" in df.columns:
        df = df[df["region"] == region]
    if plan and "plan_tier" in df.columns:
        df = df[df["plan_tier"] == plan]
    return "LLM analytics", df


def _maybe_llm_summary(title: str, fallback_summary: str, context: dict, row_count: int) -> str:
    if not llm_adapter.is_llm_enabled():
        return fallback_summary
    prompt = (
        "Rewrite this analytics summary in one concise business sentence. "
        f"Title={title}; rows={row_count}; region={context.get('region')}; plan={context.get('plan_tier')}. "
        f"Draft: {fallback_summary}"
    )
    rewritten = llm_adapter.generate_text(
        "You write concise CRM analytics summaries for business users.",
        prompt,
        temperature=0.2,
        max_output_tokens=100,
    )
    return rewritten if rewritten else fallback_summary


def run_query(user_query: str, session_id: str) -> QueryResult:
    q = user_query.strip().lower()
    context = repository.get_nl_context(session_id)

    region = _extract_region(q) or context.get("region")
    plan = _extract_plan(q) or context.get("plan_tier")

    llm_candidate = _run_llm_query(user_query, context, region, plan)
    if llm_candidate is not None:
        title, df = llm_candidate
    elif "all customers" in q or "customers" in q:
        df = repository.list_customers(region=region, plan_tier=plan)
        title = "Customer list"
    elif "open" in q and "ticket" in q:
        df = repository.to_df(
            """
            SELECT t.id, c.company_name, t.severity, t.status, t.created_at, t.subject
            FROM tickets t
            JOIN customers c ON c.id = t.customer_id
            WHERE t.status != 'Resolved'
            ORDER BY t.created_at DESC
            """
        )
        if region:
            df = df[df["company_name"].isin(repository.list_customers(region=region)["company_name"])]
        title = "Open tickets"
    elif "churn" in q or "risk" in q or "high risk" in q:
        df = repository.to_df(
            """
            SELECT c.company_name, c.region, c.plan_tier, cp.churn_probability, cp.churn_flag
            FROM churn_predictions cp
            JOIN customers c ON c.id = cp.customer_id
            JOIN (
                SELECT customer_id, MAX(id) AS latest_id
                FROM churn_predictions
                GROUP BY customer_id
            ) latest ON latest.latest_id = cp.id
            ORDER BY cp.churn_probability DESC
            """
        )
        if region:
            df = df[df["region"] == region]
        if plan:
            df = df[df["plan_tier"] == plan]
        title = "Churn risk"
    elif "health" in q:
        df = repository.to_df(
            """
            SELECT c.company_name, c.region, c.plan_tier, hs.score
            FROM health_scores hs
            JOIN customers c ON c.id = hs.customer_id
            JOIN (
                SELECT customer_id, MAX(id) AS latest_id
                FROM health_scores
                GROUP BY customer_id
            ) latest ON latest.latest_id = hs.id
            ORDER BY hs.score ASC
            """
        )
        if region:
            df = df[df["region"] == region]
        if plan:
            df = df[df["plan_tier"] == plan]
        title = "Account health"
    elif "usage" in q and "trend" in q:
        df = repository.to_df(
            """
            SELECT c.company_name, mu.month, mu.gb_used, mu.active_devices, mu.incidents
            FROM monthly_usage mu
            JOIN customers c ON c.id = mu.customer_id
            ORDER BY mu.month DESC
            """
        )
        if region:
            names = repository.list_customers(region=region)["company_name"].tolist()
            df = df[df["company_name"].isin(names)]
        title = "Usage trends"
    else:
        df = repository.list_customers(region=region, plan_tier=plan).head(20)
        title = "Fallback customer view"

    if "only" in q or "those" in q or "them" in q:
        pass

    updated_context = {
        "region": region,
        "plan_tier": plan,
        "last_intent": title,
    }
    repository.save_nl_context(session_id, updated_context)

    summary = _maybe_llm_summary(title, _build_summary(title, df, updated_context), updated_context, len(df))

    return QueryResult(title=title, summary=summary, dataframe=df, context=updated_context)


def demo_queries() -> list[str]:
    return [
        "Show all customers",
        "Show all customers in EMEA",
        "Only enterprise plans",
        "Show open tickets",
        "Show open tickets in APAC",
        "Which accounts have high churn risk?",
        "Only in North America",
        "Show account health for enterprise customers",
        "Show usage trends for APAC",
        "List customers with professional plans in LATAM",
        "Show churn risk in EMEA enterprise",
    ]
