"""
Updated app.py — integrates PII redaction before saving messages and memories.

Changes:
- Imports redact_pii and is_allowed from src.moderation.
- Adds a `metadata` column to the messages table if missing.
- Redacts incoming user messages and memories before storing them.
- Blocks saving memories that fail `is_allowed` (simple moderation heuristic).
- Stores redaction details in the messages.metadata JSON field.
"""
import json
import sqlite3
import time
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

DB_PATH = "conversations.db"

# Init DB (messages) and ensure metadata column exists
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            user TEXT,
            content TEXT,
            ts REAL
        )"""
    )
    conn.commit()
    # Ensure metadata column exists (for redaction info etc.)
    c.execute("PRAGMA table_info(messages)")
    cols = [row[1] for row in c.fetchall()]  # row[1] is column name
    if "metadata" not in cols:
        try:
            c.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
            conn.commit()
        except Exception:
            # If ALTER fails for any reason, ignore (we'll still run; older DBs without metadata will continue)
            pass
    conn.close()

init_db()

# Import Memory & OpenAI helpers
from src.memory import Memory  # ensure src/memory.py exists
from src.openai_client import call_openai_chat

# Import moderation helpers
from src.moderation import redact_pii, is_allowed

memory = Memory(db_path=DB_PATH)

app = FastAPI(title="AI_Stream — interactive chat + memory + OpenAI + redaction")

class ChatRequest(BaseModel):
    session_id: str
    user: str
    message: str

class RememberRequest(BaseModel):
    session_id: str
    user: str
    content: str
    metadata: Optional[dict] = None

def log_message(session_id: str, role: str, user: str, content: str):
    """
    Redact content before saving, store redaction metadata in messages.metadata.
    Returns the redacted content and metadata dict (useful for callers).
    """
    redacted, details = redact_pii(content)
    allowed = is_allowed(content)
    meta = {"redaction": details, "allowed": allowed}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure metadata column exists in case DB was created earlier without it
    try:
        c.execute(
            "INSERT INTO messages (session_id, role, user, content, ts, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, user, redacted, time.time(), json.dumps(meta)),
        )
    except sqlite3.OperationalError:
        # Fallback if metadata column doesn't exist (very old DB): insert without metadata
        c.execute(
            "INSERT INTO messages (session_id, role, user, content, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, user, redacted, time.time()),
        )
    conn.commit()
    conn.close()
    return redacted, meta

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="No message provided")

    # Log (and redact) user message
    redacted_msg, msg_meta = log_message(req.session_id, "user", req.user, req.message)

    # Quick explicit "remember:" handling (store memory locally rather than sending to OpenAI)
    msg_strip = req.message.strip()
    if msg_strip.lower().startswith("remember:"):
        to_remember_orig = msg_strip[len("remember:"):].strip()
        if to_remember_orig:
            # Redact memory content before storing and run allow check
            redacted_mem, mem_details = redact_pii(to_remember_orig)
            allowed = is_allowed(to_remember_orig)
            if not allowed:
                ack = "Cannot save memory: content not allowed (possible profanity or PII)."
                # Log assistant reply (redacted) and return the ack
                log_message(req.session_id, "assistant", "AI_Stream", ack)
                return {"reply": ack}
            # Save memory with metadata including redaction details
            mid = memory.add_memory(req.session_id, req.user, redacted_mem, metadata={"redaction": mem_details})
            ack = f"Saved memory id={mid}: {redacted_mem}"
            log_message(req.session_id, "assistant", "AI_Stream", ack)
            return {"reply": ack}

    # Otherwise call OpenAI with memories + recent context
    try:
        reply = await call_openai_chat(DB_PATH, memory, req.session_id, req.user, req.message)
    except Exception as e:
        # Fallback to deterministic placeholder on error but log the exception
        print("OpenAI error:", e)
        reply = f"I had trouble contacting the model. (error: {e}). Meanwhile, I heard: '{req.message}'."

    # Log assistant reply (redacted before storing; return original reply to user)
    log_message(req.session_id, "assistant", "AI_Stream", reply)
    return {"reply": reply}

@app.post("/remember")
async def remember(req: RememberRequest):
    """
    Explicit endpoint to store a memory. Redacts PII before storing.
    If content fails `is_allowed`, reject storing.
    """
    redacted, details = redact_pii(req.content)
    allowed = is_allowed(req.content)
    if not allowed:
        raise HTTPException(status_code=403, detail="Memory content not allowed (profanity/PII).")
    mid = memory.add_memory(req.session_id, req.user, redacted, metadata={"redaction": details, **(req.metadata or {})})
    # Log the explicit action (assistant ack)
    ack = f"Saved memory id={mid}: {redacted}"
    log_message(req.session_id, "assistant", "AI_Stream", ack)
    return {"status": "ok", "memory_id": mid, "redaction": details}

@app.get("/memory/search")
async def memory_search(q: str = Query(..., alias="query"), top_k: int = 5, min_sim: float = 0.3):
    results = memory.search(q, top_k=top_k, min_similarity=min_sim)
    return {"query": q, "results": results}

# WebSocket endpoint reuses the same flow (OpenAI) and redacts when storing
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_personal_message(self, session_id: str, message: str):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    First message must be JSON: {"session_id":"...", "user":"..."}
    Then subsequent messages: {"message":"..."}
    Messages starting with "remember:" will be saved (redacted) and acknowledged immediately.
    """
    try:
        init_msg = await websocket.receive_text()
        init = json.loads(init_msg)
        session_id = init.get("session_id")
        user = init.get("user", "anon")
        if not session_id:
            await websocket.close(code=4001)
            return
        await manager.connect(session_id, websocket)
        await websocket.send_text(json.dumps({"system": "connected", "session_id": session_id}))
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message")
            if not message:
                await websocket.send_text(json.dumps({"error": "no message provided"}))
                continue

            # Log (and redact) user message
            log_message(session_id, "user", user, message)

            # handle remember:
            msg_strip = message.strip()
            if msg_strip.lower().startswith("remember:"):
                to_remember_orig = msg_strip[len("remember:"):].strip()
                if to_remember_orig:
                    redacted_mem, mem_details = redact_pii(to_remember_orig)
                    allowed = is_allowed(to_remember_orig)
                    if not allowed:
                        ack = "Cannot save memory: content not allowed (possible profanity or PII)."
                        log_message(session_id, "assistant", "AI_Stream", ack)
                        await websocket.send_text(json.dumps({"reply": ack}))
                        continue
                    mid = memory.add_memory(session_id, user, redacted_mem, metadata={"redaction": mem_details})
                    ack = f"Saved memory id={mid}: {redacted_mem}"
                    log_message(session_id, "assistant", "AI_Stream", ack)
                    await websocket.send_text(json.dumps({"reply": ack}))
                    continue

            # call OpenAI
            try:
                reply = await call_openai_chat(DB_PATH, memory, session_id, user, message)
            except Exception as e:
                print("OpenAI error (ws):", e)
                reply = f"I had trouble contacting the model. (error: {e}). I heard: '{message}'."

            # Log assistant reply (redacted before storing)
            log_message(session_id, "assistant", "AI_Stream", reply)
            await websocket.send_text(json.dumps({"reply": reply}))
    except WebSocketDisconnect:
        try:
            manager.disconnect(session_id)
        except Exception:
            pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        try:
            manager.disconnect(session_id)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)