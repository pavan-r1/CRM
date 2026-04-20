from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.db import repository


_MODEL_CACHE_SIGNATURE: tuple[int, int, int] | None = None
_MODEL_CACHE_VALUE: tuple["ModelMetrics", pd.DataFrame] | None = None


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    confusion_matrix: list[list[int]]
    feature_importance: dict[str, float]


def _top_churn_factors(row: pd.Series) -> list[str]:
    candidates: list[tuple[float, str]] = [
        (
            max(0.0, (50 - float(row.get("nps_score", 50))) / 50),
            f"Low NPS: score is {int(row.get('nps_score', 50))}, indicating weaker satisfaction.",
        ),
        (
            min(float(row.get("open_tickets", 0)) / 6, 1.0),
            f"High ticket volume: {int(row.get('open_tickets', 0))} unresolved support tickets.",
        ),
        (
            min(float(row.get("critical_tickets", 0)) / 3, 1.0),
            f"Critical incidents: {int(row.get('critical_tickets', 0))} severe cases logged.",
        ),
        (
            min(float(row.get("usage_decline_pct", 0)) / 40, 1.0),
            f"Usage decline: down {float(row.get('usage_decline_pct', 0)):.1f}% over recent months.",
        ),
        (
            max(0.0, (120 - float(row.get("days_to_contract_end", 120))) / 120),
            f"Near renewal window: contract ends in {max(int(row.get('days_to_contract_end', 0)), 0)} days.",
        ),
    ]

    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    top = [message for score, message in ranked if score > 0.05][:3]
    if not top:
        top = [
            "Customer signals are stable across support, usage, and sentiment.",
            "No dominant churn driver detected from current data.",
            "Continue monitoring account trend changes monthly.",
        ]
    return top


def explain_churn_factors(top_factors_json: str) -> list[str]:
    try:
        factors = json.loads(top_factors_json)
    except json.JSONDecodeError:
        factors = []
    if not isinstance(factors, list):
        return ["No explainability factors available."]
    labels = [str(item) for item in factors][:3]
    if not labels:
        labels = ["No explainability factors available."]
    return labels


def explain_churn_reason_labels(top_factors_json: str) -> list[str]:
    factors = explain_churn_factors(top_factors_json)
    labels: list[str] = []
    for factor in factors:
        text = factor.lower()
        if "nps" in text:
            labels.append("Low NPS")
        elif "ticket" in text:
            labels.append("High ticket volume")
        elif "critical" in text:
            labels.append("Critical incidents")
        elif "usage" in text:
            labels.append("Usage decline")
        elif "renewal" in text or "contract" in text:
            labels.append("Contract renewal risk")
        else:
            labels.append("General churn risk")
    return labels[:3]


def _model_signature(df: pd.DataFrame) -> tuple[int, int, int]:
    rows = int(len(df))
    churn_sum = int(df["churn_label"].astype(int).sum()) if "churn_label" in df else 0
    id_sum = int(df["id"].astype(int).sum()) if "id" in df else 0
    return rows, churn_sum, id_sum


def build_feature_table() -> pd.DataFrame:
    customers = repository.list_customers()
    tickets = repository.to_df("SELECT * FROM tickets")
    usage = repository.to_df("SELECT * FROM monthly_usage")

    ticket_summary = tickets.groupby("customer_id").agg(
        ticket_count=("id", "count"),
        open_tickets=("status", lambda s: int((s != "Resolved").sum())),
        critical_tickets=("severity", lambda s: int((s == "Critical").sum())),
        avg_resolution_days=("resolution_days", "mean"),
    )

    usage = usage.sort_values(["customer_id", "month"])

    def usage_decline(values: pd.Series) -> float:
        if len(values) < 6:
            return 0.0
        first = values.head(6).mean()
        last = values.tail(6).mean()
        if first <= 0:
            return 0.0
        return max(0.0, float((first - last) / first * 100))

    usage_summary = usage.groupby("customer_id").agg(
        avg_gb_used=("gb_used", "mean"),
        avg_incidents=("incidents", "mean"),
        max_active_devices=("active_devices", "max"),
    )
    usage_decline_series = (
        usage.groupby("customer_id", sort=False)["gb_used"]
        .agg(usage_decline)
        .rename("usage_decline_pct")
    )
    usage_summary = usage_summary.join(usage_decline_series.to_frame(), how="left")

    features = customers.merge(ticket_summary, left_on="id", right_index=True, how="left")
    features = features.merge(usage_summary, left_on="id", right_index=True, how="left")

    features["days_to_contract_end"] = (
        pd.to_datetime(features["contract_end"]) - pd.Timestamp.today()
    ).dt.days

    features = features.fillna(0)
    return features


def train_and_score() -> tuple[ModelMetrics, pd.DataFrame]:
    global _MODEL_CACHE_SIGNATURE, _MODEL_CACHE_VALUE

    df = build_feature_table()
    signature = _model_signature(df)
    if _MODEL_CACHE_SIGNATURE == signature and _MODEL_CACHE_VALUE is not None:
        metrics, scored_cached = _MODEL_CACHE_VALUE
        return metrics, scored_cached.copy()

    feature_cols_num = [
        "nps_score",
        "ticket_count",
        "open_tickets",
        "critical_tickets",
        "avg_resolution_days",
        "avg_gb_used",
        "avg_incidents",
        "max_active_devices",
        "usage_decline_pct",
        "days_to_contract_end",
    ]
    feature_cols_cat = ["region", "plan_tier"]

    X = df[feature_cols_num + feature_cols_cat]
    y = df["churn_label"].astype(int)

    X_train, X_test, y_train, y_test, _train_ids, _test_ids = train_test_split(
        X, y, df["id"], test_size=0.25, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols_num,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols_cat,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    full_probs = model.predict_proba(X)[:, 1]

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["clf"].coef_[0]
    coef_series = pd.Series(coefficients, index=feature_names)

    readable_map = {
        "num__nps_score": "NPS",
        "num__ticket_count": "Ticket Count",
        "num__open_tickets": "Open Tickets",
        "num__critical_tickets": "Critical Tickets",
        "num__avg_resolution_days": "Resolution Time",
        "num__avg_gb_used": "Average Usage",
        "num__avg_incidents": "Incident Rate",
        "num__max_active_devices": "Active Devices",
        "num__usage_decline_pct": "Usage Decline",
        "num__days_to_contract_end": "Contract Days Remaining",
    }
    feature_importance = {
        readable_map[name]: round(float(abs(coef_series[name])), 4)
        for name in readable_map
        if name in coef_series
    }

    scored = df[["id", "company_name"]].copy()
    scored["churn_probability"] = full_probs
    scored["churn_flag"] = (scored["churn_probability"] >= 0.5).astype(int)
    scored["top_factors"] = df.apply(_top_churn_factors, axis=1)

    for idx, row in scored.iterrows():
        repository.save_churn_prediction(
            int(row["id"]),
            float(row["churn_probability"]),
            int(row["churn_flag"]),
            list(scored.loc[idx, "top_factors"]),
        )

    metrics = ModelMetrics(
        accuracy=round(float(accuracy), 3),
        precision=round(float(precision), 3),
        recall=round(float(recall), 3),
        f1=round(float(f1), 3),
        support=int((y_test == 1).sum()),
        confusion_matrix=matrix.tolist(),
        feature_importance=feature_importance,
    )
    scored_sorted = scored.sort_values("churn_probability", ascending=False)

    _MODEL_CACHE_SIGNATURE = signature
    _MODEL_CACHE_VALUE = (metrics, scored_sorted.copy())
    return metrics, scored_sorted
