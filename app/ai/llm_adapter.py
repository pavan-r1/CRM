from __future__ import annotations

import json
import os
from urllib import error, request


DEFAULT_TIMEOUT_SECONDS = 15


def _gemini_api_key() -> str:
    return str(os.getenv("GEMINI_API_KEY", "")).strip()


def _openai_api_key() -> str:
    return str(os.getenv("OPENAI_API_KEY", "")).strip()


def llm_provider() -> str:
    if _gemini_api_key():
        return "gemini"
    if _openai_api_key():
        return "openai"
    return "none"


def is_llm_enabled() -> bool:
    return llm_provider() != "none"


def _call_gemini(system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str | None:
    key = _gemini_api_key()
    if not key:
        return None

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent"
        f"?key={key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    req = request.Request(
        endpoint,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        ) or None
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return None


def _call_openai(system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str | None:
    key = _openai_api_key()
    if not key:
        return None

    endpoint = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }

    req = request.Request(
        endpoint,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )

    try:
        with request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return None


def generate_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 350,
) -> str | None:
    provider = llm_provider()
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, temperature, max_output_tokens)
    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, temperature, max_output_tokens)
    return None


def sql_from_nl(question: str, schema_hint: str, context_hint: str = "") -> str | None:
    prompt = (
        "Convert the user question into a single SQLite SELECT query. "
        "Rules: return only SQL text, no markdown, no explanation, no semicolon, no write operations.\n"
        f"Schema:\n{schema_hint}\n"
        f"Context:\n{context_hint}\n"
        f"Question:\n{question}"
    )
    return generate_text(
        "You are a SQL generator for read-only analytics.",
        prompt,
        temperature=0.0,
        max_output_tokens=260,
    )


def rewrite_query_if_enabled(user_query: str) -> str:
    if not is_llm_enabled():
        return user_query
    rewritten = generate_text(
        "You improve user analytics questions while preserving intent.",
        f"Rewrite this customer analytics question for clarity: {user_query}",
        temperature=0.1,
        max_output_tokens=120,
    )
    return rewritten if rewritten else user_query
