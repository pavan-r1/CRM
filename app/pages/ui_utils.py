from __future__ import annotations


def risk_level_from_health(score: float) -> str:
    if score >= 70:
        return "healthy"
    if score >= 45:
        return "warning"
    return "high risk"


def risk_level_from_churn(probability: float) -> str:
    if probability < 0.35:
        return "healthy"
    if probability < 0.6:
        return "warning"
    return "high risk"


def risk_color(level: str) -> str:
    if level == "healthy":
        return "#1F9D55"
    if level == "warning":
        return "#D69E2E"
    return "#D64545"


def colored_label(text: str, level: str) -> str:
    color = risk_color(level)
    return (
        f"<span style='background:{color}; color:white; padding:0.2rem 0.5rem; "
        "border-radius:0.35rem; font-weight:600; font-size:0.85rem;'>"
        f"{text}</span>"
    )
