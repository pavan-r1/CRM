from __future__ import annotations

from datetime import date

from app.db import repository


def generate_recommendations(customer_id: int, health_score: float, churn_probability: float) -> list[str]:
    customer = repository.get_customer(customer_id)
    if customer.empty:
        return ["Review customer record quality before recommending actions."]

    customer_row = customer.iloc[0]
    nps_score = int(customer_row["nps_score"])

    tickets = repository.list_tickets(customer_id)
    open_tickets = int((tickets["status"] != "Resolved").sum()) if not tickets.empty else 0
    critical_tickets = int((tickets["severity"] == "Critical").sum()) if not tickets.empty else 0

    usage = repository.to_df(
        "SELECT month, gb_used FROM monthly_usage WHERE customer_id = ? ORDER BY month",
        (customer_id,),
    )
    usage_decline = 0.0
    if len(usage) >= 6:
        first_avg = float(usage["gb_used"].head(3).mean())
        last_avg = float(usage["gb_used"].tail(3).mean())
        if first_avg > 0:
            usage_decline = max(0.0, (first_avg - last_avg) / first_avg)

    days_to_end = (date.fromisoformat(str(customer_row["contract_end"])) - date.today()).days

    actions: list[str] = []
    if churn_probability >= 0.65 or health_score < 45:
        actions.append("Schedule a customer success call within 5 business days.")
    if open_tickets >= 3 or critical_tickets >= 1:
        actions.append("Prioritize support queue and assign a named escalation owner.")
    if usage_decline >= 0.2:
        actions.append("Run an adoption workshop to recover declining product usage.")
    if nps_score < 45:
        actions.append("Send targeted feedback outreach and capture top dissatisfaction themes.")
    if days_to_end <= 90 and churn_probability >= 0.5:
        actions.append("Prepare a retention package with incentive pricing before renewal.")
    if not actions:
        actions.append("Maintain current cadence and continue monthly health monitoring.")

    if len(actions) > 4:
        return actions[:4]
    return actions
