import json
import sqlite3
import time
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = "conversations.db"

# Initialize DB
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
    conn.close()

init_db()

app = FastAPI(title="AI_Stream — interactive chat + experience logger")


class ChatRequest(BaseModel):
    session_id: str
    user: str
    message: str


def log_message(session_id: str, role: str, user: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, user, content, ts) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, user, content, time.time()),
    )
    conn.commit()
    conn.close()


def generate_reply_placeholder(session_id: str, user: str, message: str) -> str:
    """
    Replace this with a real model call (OpenAI, HF local model, etc).
    For now it's a stable placeholder that echoes and suggests next action.
    """
    # Simple deterministic reply to avoid drift during early data collection
    reply = f"I heard: '{message}'. Tell me more or ask me to remember something!"
    return reply


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="No message provided")
    # Log user message
    log_message(req.session_id, "user", req.user, req.message)
    # Get reply (swap this with a model call)
    reply = generate_reply_placeholder(req.session_id, req.user, req.message)
    # Log assistant reply
    log_message(req.session_id, "assistant", "AI_Stream", reply)
    return {"reply": reply}


# Simple WebSocket manager for streaming chat in and out
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
    Expect first message to be a JSON string: {"session_id": "...", "user": "..."}
    Then subsequent messages: {"message": "..."}
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
        # Acknowledge
        await websocket.send_text(json.dumps({"system": "connected", "session_id": session_id}))
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message")
            if not message:
                await websocket.send_text(json.dumps({"error": "no message provided"}))
                continue
            # Log user message
            log_message(session_id, "user", user, message)
            # Generate reply
            reply = generate_reply_placeholder(session_id, user, message)
            # Log assistant reply
            log_message(session_id, "assistant", "AI_Stream", reply)
            await websocket.send_text(json.dumps({"reply": reply}))
    except WebSocketDisconnect:
        # Cleanup
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