from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a script, so we ensure the project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.data.generate_synthetic_data import SeedConfig, data_quality_summary, generate_synthetic_dataset
from app.db.repository import count_customers
from app.db.schema import initialize_database
from app.pages import chatbot, customer_detail, customers, dashboard, devices, email_agent, insights, model_performance, tickets


st.set_page_config(page_title="Smart Customer Management Portal", layout="wide")


def ensure_db_ready() -> None:
    initialize_database()


def render_sidebar() -> str:
    st.sidebar.title("Smart Customer Portal")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "Overview",
            "Customer 360",
            "Customers",
            "Tickets",
            "Devices",
            "AI Chatbot",
            "AI Insights",
            "Email Agent",
            "Model Performance",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Setup")
    num_customers = st.sidebar.slider("Synthetic customers", min_value=200, max_value=600, value=220, step=10)
    if st.sidebar.button("Seed / Reseed synthetic data"):
        with st.spinner("Generating synthetic data..."):
            generate_synthetic_dataset(SeedConfig(num_customers=num_customers, random_seed=42))
        summary = data_quality_summary()
        st.sidebar.success(
            f"Loaded {summary['customers']} customers, {summary['tickets']} tickets, {summary['devices']} devices"
        )

    return page


def render_overview() -> None:
    st.title("Smart Customer Management Portal with AI-Driven Insights")
    st.write(
        "Use this demo to manage customers, tickets, and device inventory, then run health and churn analytics "
        "plus conversational natural language queries with follow-up context."
    )

    customer_count = count_customers()
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", customer_count)

    if customer_count == 0:
        st.warning("No data loaded yet. Use the sidebar to seed at least 200 synthetic customer records.")
    else:
        st.success("Data available. Use navigation to explore CRUD and AI modules.")


def main() -> None:
    ensure_db_ready()
    page = render_sidebar()

    if page == "Executive Dashboard":
        dashboard.render()
    elif page == "Overview":
        render_overview()
    elif page == "Customer 360":
        customer_detail.render()
    elif page == "Customers":
        customers.render()
    elif page == "Tickets":
        tickets.render()
    elif page == "Devices":
        devices.render()
    elif page == "AI Chatbot":
        chatbot.render()
    elif page == "AI Insights":
        insights.render()
    elif page == "Email Agent":
        email_agent.render()
    elif page == "Model Performance":
        model_performance.render()


if __name__ == "__main__":
    main()
