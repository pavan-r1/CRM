from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from app.db import repository


SEVERITY_WEIGHTS = {"Low": 1.0, "Medium": 2.0, "High": 3.5, "Critical": 5.0}


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def calculate_customer_health(customer_id: int) -> tuple[float, dict[str, float]]:
    customer = repository.get_customer(customer_id)
    if customer.empty:
        return 0.0, {"tickets": 0.0, "contract": 0.0, "usage": 0.0, "nps": 0.0}

    tickets = repository.list_tickets(customer_id)
    usage = repository.to_df(
        "SELECT * FROM monthly_usage WHERE customer_id = ? ORDER BY month", (customer_id,)
    )

    if tickets.empty:
        ticket_risk = 0.0
    else:
        ticket_risk = tickets["severity"].map(SEVERITY_WEIGHTS).mean()
        unresolved = (tickets["status"] != "Resolved").mean()
        ticket_risk = float(ticket_risk + unresolved * 1.8)

    end_date = date.fromisoformat(str(customer.iloc[0]["contract_end"]))
    days_to_end = (end_date - date.today()).days
    contract_risk = 1 - _normalize(days_to_end, 0, 540)

    usage_score = 0.5
    if len(usage) >= 6:
        first = usage["gb_used"].head(6).mean()
        last = usage["gb_used"].tail(6).mean()
        ratio = (last / first) if first > 0 else 1.0
        usage_score = float(np.clip(ratio, 0.2, 1.4) / 1.4)

    nps = float(customer.iloc[0]["nps_score"])
    nps_score = nps / 100.0

    ticket_component = 1 - _normalize(ticket_risk, 0.8, 6.5)
    contract_component = 1 - contract_risk
    usage_component = usage_score
    nps_component = nps_score

    health = (
        0.35 * ticket_component
        + 0.20 * contract_component
        + 0.25 * usage_component
        + 0.20 * nps_component
    ) * 100

    details = {
        "tickets": round(ticket_component * 100, 2),
        "contract": round(contract_component * 100, 2),
        "usage": round(usage_component * 100, 2),
        "nps": round(nps_component * 100, 2),
    }
    return round(float(np.clip(health, 0, 100)), 2), details


def refresh_all_health_scores() -> pd.DataFrame:
    customers = repository.list_customers()
    rows = []
    for customer_id in customers["id"].tolist():
        score, details = calculate_customer_health(int(customer_id))
        repository.save_health_score(int(customer_id), score, details)
        rows.append({"customer_id": int(customer_id), "score": score, "details": details})
    return pd.DataFrame(rows)


def render_health_details(details_json: str) -> str:
    details = json.loads(details_json)
    return ", ".join([f"{k}: {v:.1f}" for k, v in details.items()])


def explain_health_factors(customer_id: int, details_json: str | None = None) -> list[str]:
    if details_json is None:
        latest = repository.to_df(
            "SELECT details_json FROM health_scores WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
            (customer_id,),
        )
        if latest.empty:
            score, details = calculate_customer_health(customer_id)
            repository.save_health_score(customer_id, score, details)
            details_json = json.dumps(details)
        else:
            details_json = str(latest.iloc[0]["details_json"])

    details = json.loads(details_json)
    customer = repository.get_customer(customer_id)
    nps_score = int(customer.iloc[0]["nps_score"]) if not customer.empty else 50

    usage = repository.to_df(
        "SELECT gb_used FROM monthly_usage WHERE customer_id = ? ORDER BY month",
        (customer_id,),
    )
    usage_decline = 0.0
    if len(usage) >= 6:
        early = float(usage["gb_used"].head(3).mean())
        late = float(usage["gb_used"].tail(3).mean())
        if early > 0:
            usage_decline = max(0.0, (early - late) / early)

    tickets = repository.list_tickets(customer_id)
    open_tickets = int((tickets["status"] != "Resolved").sum()) if not tickets.empty else 0

    ranked = sorted(details.items(), key=lambda kv: float(kv[1]))
    reasons: list[str] = []
    for component, value in ranked:
        if component == "nps":
            reasons.append(f"NPS is {nps_score}, limiting customer sentiment confidence.")
        elif component == "usage":
            if usage_decline > 0:
                reasons.append(f"Usage declined by about {usage_decline:.0%} across recent months.")
            else:
                reasons.append(f"Usage contribution is moderate at {float(value):.1f}/100.")
        elif component == "tickets":
            reasons.append(f"{open_tickets} unresolved tickets are reducing the service health signal.")
        elif component == "contract":
            reasons.append("Contract timeline indicates upcoming renewal risk that needs engagement.")
        if len(reasons) == 3:
            break

    return reasons


def explain_health_reason_labels(customer_id: int, details_json: str | None = None) -> list[str]:
    reasons = explain_health_factors(customer_id, details_json)
    labels: list[str] = []
    for reason in reasons:
        text = reason.lower()
        if "nps" in text:
            labels.append("Low NPS")
        elif "usage" in text:
            labels.append("Usage decline")
        elif "ticket" in text:
            labels.append("High ticket volume")
        elif "contract" in text or "renewal" in text:
            labels.append("Contract renewal risk")
        else:
            labels.append("General account risk")
    return labels[:3]
