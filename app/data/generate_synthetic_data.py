from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from app.db import repository


REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
PLAN_TIERS = ["Starter", "Professional", "Enterprise"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
TICKET_STATUS = ["Open", "In Progress", "Resolved"]
DEVICE_TYPES = ["Router", "Switch", "Firewall", "Access Point"]
DEVICE_MODELS = ["NX-100", "NX-220", "NX-450", "EdgePro", "SecureMesh"]


@dataclass
class SeedConfig:
    num_customers: int = 220
    random_seed: int = 42


def _weighted_choice(values: list[str], weights: list[float]) -> str:
    return random.choices(values, weights=weights, k=1)[0]


def _churn_label(nps: int, usage_drop_pct: float, open_critical: int, days_to_contract_end: int) -> int:
    risk = 0.0
    if nps < 20:
        risk += 0.35
    elif nps < 40:
        risk += 0.2

    if usage_drop_pct > 35:
        risk += 0.3
    elif usage_drop_pct > 20:
        risk += 0.15

    risk += min(0.25, open_critical * 0.08)

    if days_to_contract_end <= 90:
        risk += 0.2
    elif days_to_contract_end <= 180:
        risk += 0.1

    return 1 if risk >= 0.45 else 0


def generate_synthetic_dataset(config: SeedConfig = SeedConfig()) -> None:
    fake = Faker()
    Faker.seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    customers_rows: list[dict] = []
    tickets_rows: list[dict] = []
    devices_rows: list[dict] = []
    usage_rows: list[dict] = []

    today = date.today()

    for customer_id in range(1, config.num_customers + 1):
        region = _weighted_choice(REGIONS, [0.38, 0.25, 0.25, 0.12])
        tier = _weighted_choice(PLAN_TIERS, [0.3, 0.45, 0.25])

        start_date = today - timedelta(days=random.randint(90, 1200))
        contract_duration = random.choice([365, 730, 1095])
        end_date = start_date + timedelta(days=contract_duration)

        nps_base = {
            "Starter": np.random.normal(45, 18),
            "Professional": np.random.normal(55, 16),
            "Enterprise": np.random.normal(60, 14),
        }[tier]
        nps_score = int(max(0, min(100, round(nps_base))))

        company_name = f"{fake.company()} Networks"
        email = f"ops-{customer_id}@{fake.domain_name()}"

        months = pd.date_range(end=today.replace(day=1), periods=12, freq="MS")
        usage_start = random.uniform(400, 1200) if tier != "Starter" else random.uniform(120, 500)
        trend = random.uniform(-0.12, 0.08)

        critical_open_tickets = 0
        for m_idx, month in enumerate(months):
            seasonality = 1 + 0.04 * np.sin(m_idx / 2)
            gb_used = max(10.0, usage_start * (1 + trend * m_idx) * seasonality * random.uniform(0.85, 1.15))
            active_devices = max(1, int(gb_used / random.uniform(20, 45)))
            incidents = max(0, int(np.random.poisson(2.0 if tier == "Enterprise" else 1.4)))
            usage_rows.append(
                {
                    "customer_id": customer_id,
                    "month": month.strftime("%Y-%m"),
                    "gb_used": round(float(gb_used), 2),
                    "active_devices": active_devices,
                    "incidents": incidents,
                }
            )

            tickets_this_month = np.random.poisson(2 if incidents > 1 else 1)
            for _ in range(int(tickets_this_month)):
                severity = _weighted_choice(SEVERITIES, [0.45, 0.3, 0.18, 0.07])
                status = _weighted_choice(TICKET_STATUS, [0.25, 0.2, 0.55])
                if severity == "Critical" and status != "Resolved":
                    critical_open_tickets += 1
                tickets_rows.append(
                    {
                        "customer_id": customer_id,
                        "created_at": month.strftime("%Y-%m-") + str(random.randint(1, 27)).zfill(2),
                        "severity": severity,
                        "status": status,
                        "subject": fake.sentence(nb_words=6),
                        "resolution_days": int(max(1, np.random.normal(6, 3))),
                    }
                )

        device_count = {
            "Starter": random.randint(2, 12),
            "Professional": random.randint(8, 35),
            "Enterprise": random.randint(20, 90),
        }[tier]
        for _ in range(device_count):
            devices_rows.append(
                {
                    "customer_id": customer_id,
                    "device_type": random.choice(DEVICE_TYPES),
                    "model": random.choice(DEVICE_MODELS),
                    "status": _weighted_choice(["Active", "Maintenance", "Retired"], [0.75, 0.2, 0.05]),
                    "install_date": (start_date + timedelta(days=random.randint(0, max(1, (today - start_date).days)))).isoformat(),
                }
            )

        customer_usage = [u["gb_used"] for u in usage_rows if u["customer_id"] == customer_id]
        usage_drop_pct = 0.0
        if len(customer_usage) >= 6:
            first_half = np.mean(customer_usage[:6])
            second_half = np.mean(customer_usage[6:])
            if first_half > 0:
                usage_drop_pct = max(0.0, (first_half - second_half) / first_half * 100)

        days_to_end = (end_date - today).days
        churn_label = _churn_label(nps_score, usage_drop_pct, critical_open_tickets, days_to_end)

        customers_rows.append(
            {
                "id": customer_id,
                "company_name": company_name,
                "region": region,
                "plan_tier": tier,
                "contract_start": start_date.isoformat(),
                "contract_end": end_date.isoformat(),
                "nps_score": nps_score,
                "email": email,
                "churn_label": churn_label,
            }
        )

    customers_df = pd.DataFrame(customers_rows)
    devices_df = pd.DataFrame(devices_rows)
    tickets_df = pd.DataFrame(tickets_rows)
    usage_df = pd.DataFrame(usage_rows)

    repository.replace_table("customers", customers_df)
    repository.replace_table("devices", devices_df)
    repository.replace_table("tickets", tickets_df)
    repository.replace_table("monthly_usage", usage_df)


def data_quality_summary() -> dict[str, int | float]:
    customers = repository.to_df("SELECT COUNT(*) AS c FROM customers").iloc[0]["c"]
    tickets = repository.to_df("SELECT COUNT(*) AS c FROM tickets").iloc[0]["c"]
    devices = repository.to_df("SELECT COUNT(*) AS c FROM devices").iloc[0]["c"]
    usage = repository.to_df("SELECT COUNT(*) AS c FROM monthly_usage").iloc[0]["c"]

    invalid_contracts = repository.to_df(
        "SELECT COUNT(*) AS c FROM customers WHERE contract_end <= contract_start"
    ).iloc[0]["c"]

    return {
        "customers": int(customers),
        "tickets": int(tickets),
        "devices": int(devices),
        "usage_rows": int(usage),
        "invalid_contracts": int(invalid_contracts),
    }
