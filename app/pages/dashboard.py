from __future__ import annotations

import streamlit as st

from app.ai import churn_model, health_score, nl_query
from app.ai.email_summary_agent import generate_weekly_review
from app.ai.recommendations import generate_recommendations
from app.db import repository
from app.pages.ui_utils import colored_label, risk_level_from_churn, risk_level_from_health


@st.cache_data(ttl=30)
def _cached_customers() -> object:
    return repository.list_customers()


@st.cache_data(ttl=30)
def _cached_health() -> object:
    return repository.latest_health_scores()


@st.cache_data(ttl=30)
def _cached_churn() -> object:
    return repository.latest_churn_predictions()


def _clear_dashboard_cache() -> None:
    _cached_customers.clear()
    _cached_health.clear()
    _cached_churn.clear()


def _ensure_scores_ready() -> None:
    if repository.latest_health_scores().empty:
        health_score.refresh_all_health_scores()
        _clear_dashboard_cache()
    if repository.latest_churn_predictions().empty:
        churn_model.train_and_score()
        _clear_dashboard_cache()


def _region_churn_distribution() -> None:
    churn_df = repository.latest_churn_predictions()
    customers = _cached_customers()[["id", "region"]]
    if churn_df.empty or customers.empty:
        st.info("Run model scoring to view churn distribution by region.")
        return

    merged = churn_df.merge(customers, left_on="customer_id", right_on="id", how="left")
    merged["risk_band"] = merged["churn_probability"].apply(risk_level_from_churn)
    counts = (
        merged[merged["risk_band"] == "high risk"]
        .groupby("region")
        .size()
        .rename("high_risk_customers")
        .sort_values(ascending=False)
    )
    st.bar_chart(counts)


def _monthly_usage_trend() -> None:
    usage = repository.to_df(
        """
        SELECT month, AVG(gb_used) AS avg_gb_used
        FROM monthly_usage
        GROUP BY month
        ORDER BY month
        """
    )
    if usage.empty:
        st.info("No usage data available yet.")
        return
    chart = usage.set_index("month")
    st.line_chart(chart[["avg_gb_used"]])


def _metrics_block() -> None:
    customers = _cached_customers()
    health_df = _cached_health()
    churn_df = _cached_churn()

    total_customers = len(customers)
    high_risk_customers = int((churn_df["churn_probability"] >= 0.6).sum()) if not churn_df.empty else 0
    avg_health = float(health_df["score"].mean()) if not health_df.empty else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", f"{total_customers}")
    m2.metric("High-Risk Customers", f"{high_risk_customers}")
    m3.metric("Average Health Score", f"{avg_health:.1f}")


def _business_impact() -> None:
    st.markdown("### Business Impact")
    st.markdown(
        "\n".join(
            [
                "- Detect churn early and act before account sentiment drops further.",
                "- Prioritize critical customers so success teams focus on highest-risk accounts.",
                "- Reduce manual analysis by combining model outputs, summaries, and recommendations.",
                "- Improve retention strategy with explainable signals and clear next-best actions.",
            ]
        )
    )


def _case_study_block() -> None:
    st.markdown("### Storytelling Case Study")
    case_df = repository.to_df(
        """
        SELECT c.id, c.company_name, c.region, c.plan_tier, cp.churn_probability, cp.top_factors_json
        FROM churn_predictions cp
        JOIN customers c ON c.id = cp.customer_id
        JOIN (
            SELECT customer_id, MAX(id) AS latest_id
            FROM churn_predictions
            GROUP BY customer_id
        ) latest ON latest.latest_id = cp.id
        ORDER BY cp.churn_probability DESC
        LIMIT 1
        """
    )
    if case_df.empty:
        st.info("No high-risk case study available yet. Run churn scoring first.")
        return

    row = case_df.iloc[0]
    customer_id = int(row["id"])
    reasons = churn_model.explain_churn_reason_labels(str(row["top_factors_json"]))

    health_row = repository.to_df(
        "SELECT score FROM health_scores WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
        (customer_id,),
    )
    health_value = float(health_row.iloc[0]["score"]) if not health_row.empty else 50.0

    recommendations = generate_recommendations(customer_id, health_value, float(row["churn_probability"]))
    problem_text = ", ".join(reasons[:2]) if reasons else "Multiple retention risk indicators"
    action_text = recommendations[0] if recommendations else "Schedule an account review this week."

    st.caption("A quick customer story that turns model output into a memorable business narrative.")
    st.markdown(
        f"""
        <div style='border:1px solid #E2E8F0; border-radius:12px; padding:16px; background:#F8FAFC;'>
        <div style='font-weight:700; font-size:1.02rem; margin-bottom:8px;'>{row['company_name']} ({row['region']}, {row['plan_tier']})</div>
        <div style='margin-bottom:4px;'><strong>Problem:</strong> {problem_text}</div>
        <div style='margin-bottom:4px;'><strong>AI Prediction:</strong> {float(row['churn_probability']):.1%} churn risk</div>
        <div><strong>Recommended Action:</strong> {action_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_demo_scenario() -> None:
    st.markdown("### Guided Demo Walkthrough")
    st.caption("Designed to complete in under one minute for live demos.")
    _ensure_scores_ready()

    churn_df = repository.to_df(
        """
        SELECT c.id, c.company_name, c.region, c.plan_tier, cp.churn_probability, cp.top_factors_json
        FROM churn_predictions cp
        JOIN customers c ON c.id = cp.customer_id
        JOIN (
            SELECT customer_id, MAX(id) AS latest_id
            FROM churn_predictions
            GROUP BY customer_id
        ) latest ON latest.latest_id = cp.id
        ORDER BY cp.churn_probability DESC
        LIMIT 5
        """
    )

    if churn_df.empty:
        st.warning("No churn results available. Run model training and retry the demo scenario.")
        return

    st.write("1) Highlight high-risk customers")
    churn_df["top_3_reasons"] = churn_df["top_factors_json"].apply(
        lambda value: " | ".join(churn_model.explain_churn_reason_labels(str(value)))
    )
    st.dataframe(
        churn_df[["company_name", "region", "plan_tier", "churn_probability", "top_3_reasons"]],
        width="stretch",
    )

    st.write("2) Show sample NL query result")
    demo_session = nl_query.get_or_create_session_id(st.session_state)
    query_result = nl_query.run_query("Show high risk customers in APAC enterprise", demo_session)
    st.info(query_result.summary)
    st.dataframe(query_result.dataframe.head(8), width="stretch")

    st.write("3) Open Customer 360 spotlight")
    customer_id = int(churn_df.iloc[0]["id"])
    customer = repository.get_customer(customer_id)
    if customer.empty:
        st.info("Customer 360 preview unavailable for this record.")
    else:
        customer_row = customer.iloc[0]
        latest_health = repository.to_df(
            "SELECT score, details_json FROM health_scores WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
            (customer_id,),
        )
        health_value = float(latest_health.iloc[0]["score"]) if not latest_health.empty else 0.0
        health_details = str(latest_health.iloc[0]["details_json"]) if not latest_health.empty else None
        st.markdown(
            f"{customer_row['company_name']} ({customer_row['region']}, {customer_row['plan_tier']}) | "
            f"Health {health_value:.1f} | Churn {float(churn_df.iloc[0]['churn_probability']):.1%}"
        )
        st.write(
            "Top 3 Reasons: "
            + " | ".join(health_score.explain_health_reason_labels(customer_id, health_details))
        )
        for rec in generate_recommendations(customer_id, health_value, float(churn_df.iloc[0]["churn_probability"])):
            st.write(f"- {rec}")
        st.caption("Tip: Navigate to Customer 360 and select this customer to continue the guided story.")

    st.write("4) Display email summary")
    subject, body = generate_weekly_review(customer_id)
    st.markdown(
        f"""
        <div style='border:1px solid #D1D5DB; border-radius:12px; padding:16px; background:#F8FAFC;'>
        <div style='font-weight:700; margin-bottom:8px;'>{subject}</div>
        <pre style='white-space:pre-wrap; margin:0; font-family:inherit;'>{body}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.title("Executive Dashboard")
    st.subheader("Storytelling and business impact at a glance")
    st.caption("Portfolio health and churn intelligence at a glance for leadership decisions.")

    _ensure_scores_ready()
    _metrics_block()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Region-wise High-Risk Churn Distribution")
        _region_churn_distribution()
    with col2:
        st.markdown("#### Monthly Usage Trend")
        _monthly_usage_trend()

    health_df = _cached_health()
    churn_df = _cached_churn()
    if not health_df.empty and not churn_df.empty:
        merged = churn_df.merge(health_df[["customer_id", "score"]], on="customer_id", how="left")
        merged["health_risk"] = merged["score"].apply(risk_level_from_health)
        merged["churn_risk"] = merged["churn_probability"].apply(risk_level_from_churn)
        high_risk = merged.sort_values(["churn_probability", "score"], ascending=[False, True]).head(8)
        st.markdown("### Priority Accounts")
        display = high_risk[["customer_id", "score", "churn_probability", "health_risk", "churn_risk"]].copy()
        st.dataframe(display, width="stretch", hide_index=True)
        st.markdown(
            f"Health status: {colored_label('Healthy', 'healthy')} {colored_label('Warning', 'warning')} {colored_label('High Risk', 'high risk')}",
            unsafe_allow_html=True,
        )
    else:
        st.info("Run health and churn scoring to populate priority account insights.")

    st.markdown("---")
    _case_study_block()
    st.markdown("---")

    _business_impact()

    if st.button("Run Demo Scenario"):
        try:
            _run_demo_scenario()
        except Exception as exc:
            st.error(f"Demo scenario hit an unexpected issue: {exc}")

    st.markdown("### Final Impact")
    st.success(
        "This system enables proactive decision-making and helps reduce customer churn by identifying risks early."
    )
