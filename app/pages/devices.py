from __future__ import annotations

import streamlit as st

from app.db import repository


DEVICE_TYPES = ["Router", "Switch", "Firewall", "Access Point"]
DEVICE_STATUS = ["Active", "Maintenance", "Retired"]


def render() -> None:
    st.subheader("Device Inventory")

    customers = repository.list_customers()
    if customers.empty:
        st.info("No customers available. Seed data first.")
        return

    options = {f"{int(row['id'])} - {row['company_name']}": int(row["id"]) for _, row in customers.iterrows()}

    with st.expander("Create or Update Device", expanded=False):
        with st.form("device_form"):
            device_id = st.number_input("Device ID (0 for new)", min_value=0, step=1)
            customer_key = st.selectbox("Customer", list(options.keys()))
            device_type = st.selectbox("Device type", DEVICE_TYPES)
            model = st.text_input("Model")
            status = st.selectbox("Status", DEVICE_STATUS)
            install_date = st.date_input("Install date")
            submitted = st.form_submit_button("Save device")
            if submitted:
                repository.upsert_device(
                    {
                        "id": int(device_id) if device_id > 0 else None,
                        "customer_id": options[customer_key],
                        "device_type": device_type,
                        "model": model,
                        "status": status,
                        "install_date": install_date.isoformat(),
                    }
                )
                st.success("Device saved.")

    with st.expander("Delete Device", expanded=False):
        delete_id = st.number_input("Device ID to delete", min_value=1, step=1)
        if st.button("Delete device"):
            repository.delete_device(int(delete_id))
            st.warning(f"Deleted device {delete_id}.")

    st.dataframe(repository.list_devices(), width="stretch", hide_index=True)
