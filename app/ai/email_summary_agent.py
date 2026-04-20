from __future__ import annotations

from datetime import date

from app.ai import llm_adapter
from app.db import repository


def generate_weekly_review(customer_id: int) -> tuple[str, str]:
    customer = repository.get_customer(customer_id)
    if customer.empty:
        return "Customer Not Found", "No record available for this customer id."

    c = customer.iloc[0]
    company = c["company_name"]

    health = repository.to_df(
        "SELECT score FROM health_scores WHERE customer_id=? ORDER BY id DESC LIMIT 1", (customer_id,)
    )
    churn = repository.to_df(
        "SELECT churn_probability, churn_flag FROM churn_predictions WHERE customer_id=? ORDER BY id DESC LIMIT 1",
        (customer_id,),
    )
    tickets = repository.to_df(
        """
        SELECT severity, status, COUNT(*) AS cnt
        FROM tickets
        WHERE customer_id=? AND created_at >= date('now', '-30 day')
        GROUP BY severity, status
        ORDER BY cnt DESC
        """,
        (customer_id,),
    )
    usage = repository.to_df(
        "SELECT month, gb_used FROM monthly_usage WHERE customer_id=? ORDER BY month DESC LIMIT 3",
        (customer_id,),
    )

    health_score = float(health.iloc[0]["score"]) if not health.empty else 0.0
    churn_prob = float(churn.iloc[0]["churn_probability"]) if not churn.empty else 0.0
    churn_flag = int(churn.iloc[0]["churn_flag"]) if not churn.empty else 0

    usage_line = "No recent usage data available."
    if len(usage) >= 2:
        latest = float(usage.iloc[0]["gb_used"])
        prev = float(usage.iloc[1]["gb_used"])
        delta = latest - prev
        direction = "up" if delta >= 0 else "down"
        usage_line = f"Usage is {direction} by {abs(delta):.1f} GB compared with previous month."

    ticket_summary = "No recent tickets in the last 30 days."
    if not tickets.empty:
        top = tickets.head(3)
        ticket_summary = "; ".join(
            [f"{int(r['cnt'])} {r['severity']} ({r['status']})" for _, r in top.iterrows()]
        )

    recommendations = []
    if health_score < 55:
        recommendations.append("Escalate an account success review within 7 days.")
    if churn_prob > 0.6:
        recommendations.append("Prepare a retention offer prior to renewal discussion.")
    if not recommendations:
        recommendations.append("Maintain cadence and monitor trends next week.")

    subject = f"Weekly Account Review: {company} ({date.today().isoformat()})"
    body = (
        f"Account: {company}\n"
        f"Region: {c['region']} | Plan: {c['plan_tier']}\n"
        f"Health Score: {health_score:.1f}/100\n"
        f"Churn Probability (90d): {churn_prob:.2%} | Flag: {'High' if churn_flag else 'Low'}\n\n"
        f"Ticket Highlights (last 30 days): {ticket_summary}\n"
        f"Usage Summary: {usage_line}\n\n"
        "Recommended Actions:\n"
        + "\n".join([f"- {item}" for item in recommendations])
    )

    if llm_adapter.is_llm_enabled():
        rewrite_prompt = (
            "Rewrite this weekly account review email to sound professional, executive-friendly, "
            "and concise while preserving all factual values and recommendations.\n\n"
            f"{body}"
        )
        rewritten = llm_adapter.generate_text(
            "You are a B2B customer success email assistant.",
            rewrite_prompt,
            temperature=0.2,
            max_output_tokens=500,
        )
        if rewritten:
            body = rewritten

    return subject, body
