from __future__ import annotations

import json

import plotly.express as px
import streamlit as st

from app.ai import churn_model, health_score, llm_adapter, nl_query
from app.db import repository


def _rule_based_insight(question: str) -> str:
    churn = repository.to_df(
        """
        SELECT c.region, c.plan_tier, cp.churn_probability
        FROM churn_predictions cp
        JOIN customers c ON c.id = cp.customer_id
        JOIN (
            SELECT customer_id, MAX(id) AS latest_id
            FROM churn_predictions
            GROUP BY customer_id
        ) latest ON latest.latest_id = cp.id
        """
    )
    if churn.empty:
        return "No churn predictions are available yet. Run churn training first."

    high = churn[churn["churn_probability"] >= 0.6]
    if high.empty:
        return "Current data does not show a significant high-risk churn cluster."

    top_region = high.groupby("region").size().sort_values(ascending=False).index[0]
    enterprise_share = float((high["plan_tier"] == "Enterprise").mean())
    return (
        f"High churn is concentrated in {top_region}. "
        f"About {enterprise_share:.0%} of high-risk accounts are enterprise, suggesting targeted success outreach there."
    )


def _llm_analytical_insight(question: str) -> str:
    if not llm_adapter.is_llm_enabled():
        return _rule_based_insight(question)

    churn = repository.to_df(
        """
        SELECT c.company_name, c.region, c.plan_tier, cp.churn_probability
        FROM churn_predictions cp
        JOIN customers c ON c.id = cp.customer_id
        JOIN (
            SELECT customer_id, MAX(id) AS latest_id
            FROM churn_predictions
            GROUP BY customer_id
        ) latest ON latest.latest_id = cp.id
        ORDER BY cp.churn_probability DESC
        LIMIT 80
        """
    )
    usage = repository.to_df(
        """
        SELECT c.region, AVG(mu.gb_used) AS avg_usage, AVG(mu.incidents) AS avg_incidents
        FROM monthly_usage mu
        JOIN customers c ON c.id = mu.customer_id
        GROUP BY c.region
        """
    )

    prompt = (
        "Answer as a CRM analyst in 3-4 sentences. Mention likely drivers and recommended action.\n"
        f"Question: {question}\n"
        f"Churn sample: {churn.head(20).to_dict(orient='records')}\n"
        f"Usage by region: {usage.to_dict(orient='records')}"
    )
    generated = llm_adapter.generate_text(
        "You are a customer retention analytics assistant.",
        prompt,
        temperature=0.2,
        max_output_tokens=280,
    )
    return generated if generated else _rule_based_insight(question)


def render() -> None:
    st.subheader("AI Insights")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Account Health Scores"):
            df = health_score.refresh_all_health_scores()
            st.success(f"Updated health scores for {len(df)} customers.")

    with col2:
        if st.button("Run Churn Prediction Training"):
            metrics, scored = churn_model.train_and_score()
            st.success(
                f"Model metrics - Precision: {metrics.precision}, Recall: {metrics.recall}, F1: {metrics.f1}"
            )
            st.dataframe(scored.head(20), width="stretch", hide_index=True)

    health_df = repository.latest_health_scores()
    churn_df = repository.latest_churn_predictions()

    if not health_df.empty:
        merged = health_df.merge(repository.list_customers()[["id", "company_name"]], left_on="customer_id", right_on="id")
        merged = merged.sort_values("score", ascending=True).head(20)
        merged["top_3_reasons"] = merged.apply(
            lambda row: " | ".join(health_score.explain_health_factors(int(row["customer_id"]), str(row["details_json"]))),
            axis=1,
        )
        st.markdown("#### Lowest 20 Health Scores")
        fig = px.bar(merged, x="company_name", y="score", hover_data=["top_3_reasons"])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(merged[["company_name", "score", "top_3_reasons"]], width="stretch", hide_index=True)

    if not churn_df.empty:
        churn_merged = churn_df.merge(repository.list_customers()[["id", "company_name"]], left_on="customer_id", right_on="id")
        churn_merged = churn_merged.sort_values("churn_probability", ascending=False).head(20)
        churn_merged["top_3_reasons"] = churn_merged["top_factors_json"].apply(
            lambda v: " | ".join(churn_model.explain_churn_factors(str(v)))
        )
        st.markdown("#### Top 20 Churn Risks")
        fig = px.bar(churn_merged, x="company_name", y="churn_probability", hover_data=["top_3_reasons"])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            churn_merged[["company_name", "churn_probability", "top_3_reasons"]],
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Natural Language Query")
    session_id = nl_query.get_or_create_session_id(st.session_state)
    prompt = st.text_input("Ask a question about customers, tickets, health, churn, or usage")
    if st.button("Run query") and prompt.strip():
        result = nl_query.run_query(prompt, session_id)
        st.markdown(f"**{result.title}**")
        st.info(result.summary)
        st.dataframe(result.dataframe, width="stretch", hide_index=True)
        st.caption(f"Context: {json.dumps(result.context)}")

    st.markdown("### AI Analyst")
    analyst_question = st.text_input("Ask an analytical question", placeholder="Why is churn high in APAC?")
    if st.button("Generate AI Insight") and analyst_question.strip():
        st.info(_llm_analytical_insight(analyst_question))

    with st.expander("Demo query list"):
        for idx, q in enumerate(nl_query.demo_queries(), start=1):
            st.write(f"{idx}. {q}")
