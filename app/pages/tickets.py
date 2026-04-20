from __future__ import annotations

import streamlit as st

from app.db import repository


SEVERITY = ["Low", "Medium", "High", "Critical"]
STATUS = ["Open", "In Progress", "Resolved"]


def render() -> None:
    st.subheader("Tickets")

    customers = repository.list_customers()
    if customers.empty:
        st.info("No customers available. Seed data first.")
        return

    options = {f"{int(row['id'])} - {row['company_name']}": int(row["id"]) for _, row in customers.iterrows()}

    with st.expander("Create or Update Ticket", expanded=False):
        with st.form("ticket_form"):
            ticket_id = st.number_input("Ticket ID (0 for new)", min_value=0, step=1)
            customer_key = st.selectbox("Customer", list(options.keys()))
            created_at = st.date_input("Created date")
            severity = st.selectbox("Severity", SEVERITY)
            status = st.selectbox("Status", STATUS)
            subject = st.text_input("Subject")
            resolution_days = st.number_input("Resolution days", min_value=1, value=5, step=1)
            submitted = st.form_submit_button("Save ticket")
            if submitted:
                repository.upsert_ticket(
                    {
                        "id": int(ticket_id) if ticket_id > 0 else None,
                        "customer_id": options[customer_key],
                        "created_at": created_at.isoformat(),
                        "severity": severity,
                        "status": status,
                        "subject": subject,
                        "resolution_days": int(resolution_days),
                    }
                )
                st.success("Ticket saved.")

    with st.expander("Delete Ticket", expanded=False):
        delete_id = st.number_input("Ticket ID to delete", min_value=1, step=1)
        if st.button("Delete ticket"):
            repository.delete_ticket(int(delete_id))
            st.warning(f"Deleted ticket {delete_id}.")

    st.dataframe(repository.list_tickets(), width="stretch", hide_index=True)
