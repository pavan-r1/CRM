from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from app.ai import llm_adapter, nl_query
from app.db import repository


KEYWORDS = (
    "churn",
    "risk",
    "health",
    "ticket",
    "tickets",
    "usage",
    "customer",
    "customers",
    "region",
    "apac",
    "emea",
    "latam",
    "north america",
    "enterprise",
    "starter",
    "professional",
)


def _default_chat_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "I can answer customer health, churn, tickets, usage, and retention questions. "
                "Ask me things like: 'Why is churn high in APAC?' or 'Show me high-risk enterprise accounts.'"
            ),
        }
    ]


def ensure_chat_state() -> None:
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = _default_chat_history()
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    if "chat_prompt" not in st.session_state:
        st.session_state.chat_prompt = ""


@st.cache_data(ttl=60)
def _portfolio_snapshot() -> dict:
    customers = repository.list_customers()
    health_df = repository.latest_health_scores()
    churn_df = repository.latest_churn_predictions()

    total_customers = int(len(customers))
    high_risk_customers = int((churn_df["churn_probability"] >= 0.6).sum()) if not churn_df.empty else 0
    avg_health = float(health_df["score"].mean()) if not health_df.empty else 0.0

    region_summary = "No data loaded yet."
    if not customers.empty and "region" in customers.columns:
        region_counts = customers["region"].value_counts().head(3)
        region_summary = ", ".join([f"{region}: {int(count)}" for region, count in region_counts.items()])

    return {
        "total_customers": total_customers,
        "high_risk_customers": high_risk_customers,
        "avg_health": avg_health,
        "region_summary": region_summary,
    }


def _initial_messages() -> list[dict[str, str]]:
    return _default_chat_history()


def _is_analytics_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in KEYWORDS) or q.startswith(("show", "list", "why", "which", "what"))


def _recent_history(messages: list[dict[str, str]]) -> str:
    recent = messages[-6:]
    return "\n".join([f"{item['role']}: {item['content']}" for item in recent])


def _format_dataframe_preview(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows matched the request."
    preview = df.head(5).copy()
    return preview.to_string(index=False)


def _answer_with_rules(question: str, session_id: str, messages: list[dict[str, str]]) -> str:
    if _is_analytics_question(question):
        result = nl_query.run_query(question, session_id)
        return f"{result.summary}\n\nTop results:\n{_format_dataframe_preview(result.dataframe)}"

    snapshot = _portfolio_snapshot()
    lower = question.lower()
    if "help" in lower or "can you do" in lower:
        return (
            "I can help analyze churn, health, tickets, usage, and customer segments. "
            "Try asking for a region, plan tier, or risk level, and I will summarize it for you."
        )
    if "high risk" in lower or "risk" in lower:
        return (
            f"There are {snapshot['high_risk_customers']} high-risk customers out of {snapshot['total_customers']} total."
        )
    if "health" in lower:
        return f"Average health score is {snapshot['avg_health']:.1f}."
    return (
        f"Current portfolio snapshot: {snapshot['total_customers']} customers, "
        f"{snapshot['high_risk_customers']} high-risk accounts, top regions: {snapshot['region_summary']}."
    )


def _answer_with_llm(question: str, session_id: str, messages: list[dict[str, str]]) -> str | None:
    snapshot = _portfolio_snapshot()
    history = _recent_history(messages)
    base_answer = _answer_with_rules(question, session_id, messages)
    prompt = (
        "You are an AI chatbot for a Smart Customer Management Portal. "
        "Keep answers concise, practical, and business-focused. Use the portfolio snapshot and recent conversation. "
        "If the user asks for analytics, explain the result in simple business language and mention next steps.\n\n"
        f"Portfolio snapshot: {snapshot}\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Draft answer to improve:\n{base_answer}\n\n"
        f"User question: {question}"
    )
    return llm_adapter.generate_text(
        "You are a helpful CRM copilot for customer success teams.",
        prompt,
        temperature=0.2,
        max_output_tokens=220,
    )


def _get_response(question: str, session_id: str, messages: list[dict[str, str]]) -> str:
    if llm_adapter.is_llm_enabled():
        response = _answer_with_llm(question, session_id, messages)
        if response:
            return response
    return _answer_with_rules(question, session_id, messages)


def _reset_chat() -> None:
    st.session_state.chat_history = _default_chat_history()
    st.session_state.chat_session_id = str(uuid.uuid4())
    st.session_state.chat_prompt = ""


def _append_user_message(question: str) -> str:
    ensure_chat_state()
    st.session_state.chat_history.append({"role": "user", "content": question})
    response = _get_response(question, st.session_state.chat_session_id, st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    return response


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def _render_suggested_questions(key_prefix: str) -> None:
    quick_prompts = [
        "Show high-risk customers",
        "Why is churn increasing?",
        "Top customers with low NPS",
    ]
    quick_columns = st.columns(len(quick_prompts))
    for index, quick_prompt in enumerate(quick_prompts):
        with quick_columns[index]:
            if st.button(quick_prompt, key=f"{key_prefix}_prompt_{index}"):
                _append_user_message(quick_prompt)
                st.rerun()


def _render_chat_input(key_prefix: str) -> None:
    prompt_key = f"{key_prefix}_prompt"
    question = st.text_input(
        "Ask a question",
        key=prompt_key,
        placeholder="Ask about churn, health, tickets, or usage",
    )
    if st.button("Send", key=f"{key_prefix}_send", type="secondary") and question.strip():
        st.session_state[prompt_key] = ""
        _append_user_message(question.strip())
        st.rerun()


def _render_panel_header(key_prefix: str) -> None:
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown("### AI Assistant")
    with header_right:
        if st.button("Close", key=f"{key_prefix}_close"):
            st.session_state.show_chat = False
            st.rerun()


def render_floating_widget() -> None:
    ensure_chat_state()

    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button[kind="primary"] {
            position: fixed;
            right: 20px;
            bottom: 20px;
            z-index: 10000;
            border-radius: 999px;
            padding: 0.7rem 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            position: fixed;
            right: 20px;
            bottom: 78px;
            width: 380px;
            max-width: calc(100vw - 40px);
            max-height: 72vh;
            overflow-y: auto;
            z-index: 9999;
            background: white;
            border-radius: 16px;
            box-shadow: 0 14px 40px rgba(0,0,0,0.20);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("💬 AI", key="floating_chat_toggle", type="primary"):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

    if not st.session_state.show_chat:
        return

    with st.container(border=True):
        _render_panel_header("floating_chat")
        st.caption("Quick answers for churn, health, tickets, usage, and retention.")
        _render_suggested_questions("floating_chat")
        st.divider()
        _render_chat_history()
        _render_chat_input("floating_chat")


def render() -> None:
    st.subheader("AI Chatbot")
    st.caption("A simple customer success copilot for live questions, summaries, and follow-up context.")

    ensure_chat_state()
    if "chat_messages" in st.session_state and not st.session_state.chat_history:
        st.session_state.chat_history = st.session_state.chat_messages
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = st.session_state.chat_history

    top_left, top_right = st.columns([3, 1])
    with top_right:
        if st.button("Reset chat"):
            _reset_chat()
            st.rerun()

    snapshot = _portfolio_snapshot()
    st.markdown(
        f"**Portfolio:** {snapshot['total_customers']} customers | "
        f"{snapshot['high_risk_customers']} high-risk | Avg health {snapshot['avg_health']:.1f}"
    )

    _render_chat_history()
    _render_suggested_questions("page_chat")
    st.divider()
    _render_chat_input("page_chat")
