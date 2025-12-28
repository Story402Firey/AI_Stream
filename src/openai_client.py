"""
"""
import os
import time
import sqlite3
import asyncio
from typing import List, Dict, Any, Optional
import logging

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("ai_stream.openai_client")

# Configure OpenAI key if present in environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY not set; OpenAI calls will fail until API key is provided.")

def _fetch_recent_messages(db_path: str, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT role, user, content, ts FROM messages WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
        (session_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    rows = list(reversed(rows))
    msgs = [{"role": r[0], "user": r[1], "content": r[2], "ts": r[3]} for r in rows]
    return msgs

def _build_system_prompt(mems: List[Dict[str, Any]]) -> str:
    base = (
        "You are AI_Stream, a friendly, helpful assistant used on live streams. "
        "Use the provided memory facts about viewers to personalize answers when appropriate. "
        "Be concise and avoid revealing private data. If a memory looks like personal data (PII), do not repeat it verbatim — summarize or ask for permission."
    )
    if not mems:
        return base + "\n\nNo relevant memories available."
    mem_lines = []
    for m in mems:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("ts", time.time())))
        mem_lines.append(f"- ({m.get('similarity', 0):.2f}) {m.get('user')}: {m.get('content')} [{ts}]")
    mem_block = "Relevant memories (most similar first):\n" + "\n".join(mem_lines)
    return base + "\n\n" + mem_block

# Retry configuration: 3 attempts, exponential backoff (wait 1s, 2s, 4s...), retry on network / API errors
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(Exception))
def _sync_openai_call(model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int):
    """
    Synchronous OpenAI call wrapped with tenacity retry logic.
    Called inside a thread via asyncio.to_thread.
    """
    logger.debug("Calling OpenAI model=%s messages_len=%d temperature=%s max_tokens=%s", model, len(messages), temperature, max_tokens)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp

async def call_openai_chat(
    db_path: str,
    memory_obj,
    session_id: str,
    user: str,
    message: str,
    top_k_mems: int = 3,
    mem_min_sim: float = 0.25,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set. See .env.sample")

    model = model or OPENAI_MODEL
    temperature = OPENAI_TEMPERATURE if temperature is None else temperature
    max_tokens = OPENAI_MAX_TOKENS if max_tokens is None else max_tokens

    # 1) Fetch top memories
    try:
        mems = memory_obj.search(message, top_k=top_k_mems, min_similarity=mem_min_sim)
    except Exception as e:
        logger.exception("Memory search failed: %s", e)
        mems = []

    system_prompt = _build_system_prompt(mems)

    # 2) Fetch recent conversation context
    recent = _fetch_recent_messages(db_path, session_id, limit=8)

    # 3) Assemble messages for OpenAI (system message + recent context + user message)
    msgs = [{"role": "system", "content": system_prompt}]

    for m in recent:
        role = "user" if m["role"] == "user" else "assistant"
        content = f"{m['user']}: {m['content']}" if role == "user" else m["content"]
        msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": message})

    # 4) Call OpenAI in thread to avoid blocking
    try:
        resp = await asyncio.to_thread(_sync_openai_call, model, msgs, temperature, max_tokens)
        reply = ""
        if getattr(resp, "choices", None):
            reply = resp.choices[0].message.content
        elif isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                reply = msg.get("content", "")
        return (reply or "").strip()
    except Exception as e:
        logger.exception("OpenAI request failed after retries: %s", e)
        raise RuntimeError(f"OpenAI request failed: {e}")
