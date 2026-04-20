from __future__ import annotations

import streamlit as st

from app.db import repository


REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
TIERS = ["Starter", "Professional", "Enterprise"]


def render() -> None:
    st.subheader("Customers")

    with st.expander("Create or Update Customer", expanded=False):
        with st.form("customer_form"):
            customer_id = st.number_input("Customer ID (0 for new)", min_value=0, step=1)
            col1, col2 = st.columns(2)
            with col1:
                company_name = st.text_input("Company name")
                region = st.selectbox("Region", REGIONS)
                plan_tier = st.selectbox("Plan tier", TIERS)
                nps_score = st.slider("NPS", min_value=0, max_value=100, value=50)
            with col2:
                contract_start = st.date_input("Contract start")
                contract_end = st.date_input("Contract end")
                email = st.text_input("Account email")
                churn_label = st.selectbox("Synthetic churn label", [0, 1])

            submitted = st.form_submit_button("Save customer")
            if submitted:
                if contract_end <= contract_start:
                    st.error("Contract end date must be after start date.")
                else:
                    repository.upsert_customer(
                        {
                            "id": int(customer_id) if customer_id > 0 else None,
                            "company_name": company_name,
                            "region": region,
                            "plan_tier": plan_tier,
                            "contract_start": contract_start.isoformat(),
                            "contract_end": contract_end.isoformat(),
                            "nps_score": nps_score,
                            "email": email,
                            "churn_label": churn_label,
                        }
                    )
                    st.success("Customer saved.")

    with st.expander("Delete Customer", expanded=False):
        delete_id = st.number_input("Customer ID to delete", min_value=1, step=1)
        if st.button("Delete customer"):
            repository.delete_customer(int(delete_id))
            st.warning(f"Deleted customer {delete_id}.")

    region_filter = st.selectbox("Filter by region", ["All"] + REGIONS)
    tier_filter = st.selectbox("Filter by plan", ["All"] + TIERS)
    region = None if region_filter == "All" else region_filter
    tier = None if tier_filter == "All" else tier_filter

    df = repository.list_customers(region=region, plan_tier=tier)
    st.dataframe(df, width="stretch", hide_index=True)
