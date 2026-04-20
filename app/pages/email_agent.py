from __future__ import annotations

import streamlit as st

from app.ai.email_summary_agent import generate_weekly_review
from app.ai.llm_adapter import is_llm_enabled, llm_provider
from app.db import repository


def render() -> None:
    st.subheader("Weekly Account Review Email Agent")
    mode = f"LLM mode: {llm_provider()}" if is_llm_enabled() else "LLM mode: fallback template"
    st.caption(mode)

    customers = repository.list_customers()
    if customers.empty:
        st.info("No customers available. Seed data first.")
        return

    options = {f"{int(row['id'])} - {row['company_name']}": int(row["id"]) for _, row in customers.iterrows()}
    selected = st.selectbox("Customer", list(options.keys()))

    if st.button("Generate weekly summary"):
        customer_id = options[selected]
        subject, body = generate_weekly_review(customer_id)
        repository.save_generated_email(customer_id, subject, body)
        st.markdown("### Generated Email")
        st.markdown(
            f"""
            <div style='border:1px solid #D1D5DB; border-radius:12px; padding:16px; background:#F8FAFC;'>
            <div style='font-weight:700; margin-bottom:10px;'>Subject: {subject}</div>
            <pre style='white-space:pre-wrap; margin:0; font-family:inherit;'>{body}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(f"Subject: {subject}\n\n{body}", language="markdown")
        st.caption("Use the copy icon in the code block to quickly copy the email content.")
