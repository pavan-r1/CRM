from __future__ import annotations

import streamlit as st

from app.ai import churn_model, health_score, llm_adapter
from app.ai.recommendations import generate_recommendations
from app.db import repository
from app.pages.ui_utils import colored_label, risk_level_from_churn, risk_level_from_health


def _latest_health(customer_id: int) -> tuple[float, str]:
    latest = repository.to_df(
        "SELECT score, details_json FROM health_scores WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
        (customer_id,),
    )
    if latest.empty:
        score, details = health_score.calculate_customer_health(customer_id)
        repository.save_health_score(customer_id, score, details)
        latest = repository.to_df(
            "SELECT score, details_json FROM health_scores WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
            (customer_id,),
        )
    return float(latest.iloc[0]["score"]), str(latest.iloc[0]["details_json"])


def _latest_churn(customer_id: int) -> tuple[float, str]:
    latest = repository.to_df(
        """
        SELECT churn_probability, top_factors_json
        FROM churn_predictions
        WHERE customer_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (customer_id,),
    )
    if latest.empty:
        churn_model.train_and_score()
        latest = repository.to_df(
            """
            SELECT churn_probability, top_factors_json
            FROM churn_predictions
            WHERE customer_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (customer_id,),
        )
    return float(latest.iloc[0]["churn_probability"]), str(latest.iloc[0]["top_factors_json"])


def render() -> None:
    st.subheader("Customer 360 View")
    customers = repository.list_customers()
    if customers.empty:
        st.info("No customers found. Seed synthetic data from the sidebar first.")
        return

    options = {f"{int(row['id'])} - {row['company_name']}": int(row["id"]) for _, row in customers.iterrows()}
    selected = st.selectbox("Select customer", list(options.keys()))
    customer_id = options[selected]

    profile = repository.get_customer(customer_id).iloc[0]
    health_value, details_json = _latest_health(customer_id)
    churn_probability, churn_factors_json = _latest_churn(customer_id)

    health_level = risk_level_from_health(health_value)
    churn_level = risk_level_from_churn(churn_probability)

    st.markdown(
        f"{colored_label('Health: ' + health_level.title(), health_level)} "
        f"{colored_label('Churn: ' + churn_level.title(), churn_level)}",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Health Score", f"{health_value:.1f}")
    c2.metric("Churn Probability", f"{churn_probability:.1%}")
    c3.metric("NPS", f"{int(profile['nps_score'])}")

    st.markdown("### Profile")
    st.write(
        {
            "company_name": profile["company_name"],
            "region": profile["region"],
            "plan_tier": profile["plan_tier"],
            "contract_end": profile["contract_end"],
            "email": profile["email"],
        }
    )

    st.markdown("### Explainable AI")
    health_reasons = health_score.explain_health_factors(customer_id, details_json)
    health_labels = health_score.explain_health_reason_labels(customer_id, details_json)
    churn_reasons = churn_model.explain_churn_factors(churn_factors_json)
    churn_labels = churn_model.explain_churn_reason_labels(churn_factors_json)

    if churn_probability >= 0.6:
        risk_phrase = "high risk"
    elif churn_probability >= 0.35:
        risk_phrase = "moderate risk"
    else:
        risk_phrase = "low risk"

    dominant_reasons = churn_labels[:2] if churn_labels else ["general risk signals"]
    reason_text = " and ".join(dominant_reasons)
    st.info(f"AI Explanation: This customer is at {risk_phrase} due to {reason_text}.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Health Top 3 Reasons")
        st.write(" | ".join(health_labels))
        for reason in health_reasons:
            st.write(f"- {reason}")
    with right:
        st.markdown("#### Churn Top 3 Reasons")
        st.write(" | ".join(churn_labels))
        for reason in churn_reasons:
            st.write(f"- {reason}")

    if llm_adapter.is_llm_enabled():
        explanation_prompt = (
            f"Customer region={profile['region']}, plan={profile['plan_tier']}, nps={int(profile['nps_score'])}, "
            f"health={health_value:.1f}, churn={churn_probability:.2f}, "
            f"health_reasons={health_reasons}, churn_reasons={churn_reasons}. "
            "Write a concise business interpretation and one immediate action."
        )
        narrative = llm_adapter.generate_text(
            "You are an enterprise customer success strategist.",
            explanation_prompt,
            temperature=0.2,
            max_output_tokens=180,
        )
        if narrative:
            st.markdown("#### Advanced AI Interpretation")
            st.info(narrative)
    else:
        st.caption("Advanced interpretation is in fallback mode (no LLM key detected).")

    st.markdown("### Ticket History")
    tickets = repository.list_tickets(customer_id)
    if tickets.empty:
        st.info("No ticket history for this customer.")
    else:
        severity_counts = tickets.groupby("severity").size().rename("ticket_count")
        st.bar_chart(severity_counts)
        st.dataframe(tickets[["created_at", "severity", "status", "subject"]], width="stretch", hide_index=True)

    st.markdown("### Usage Trend")
    usage = repository.to_df(
        "SELECT month, gb_used, active_devices, incidents FROM monthly_usage WHERE customer_id = ? ORDER BY month",
        (customer_id,),
    )
    if usage.empty:
        st.info("No usage history available.")
    else:
        trend = usage.set_index("month")
        st.line_chart(trend[["gb_used"]])

    st.markdown("### AI Recommendations")
    for recommendation in generate_recommendations(customer_id, health_value, churn_probability):
        st.write(f"- {recommendation}")
