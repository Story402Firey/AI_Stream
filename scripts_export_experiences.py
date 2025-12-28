#!/usr/bin/env python3
"""
Aggregate conversations from SQLite and export supervised training JSONL.

Rules:
- For each session, pair user message -> assistant reply as a training example.
- Output: data/experiences.jsonl with {"prompt": "<user message>", "response": "<assistant reply>"}
  (you can extend prompt to include context).
"""
import sqlite3
import json
import argparse
from pathlib import Path

DB_PATH = "conversations.db"


def export_jsonl(out_path: str, min_pairs: int = 1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Read messages grouped by session ordered by timestamp
    c.execute("SELECT session_id, role, user, content, ts FROM messages ORDER BY session_id, ts")
    rows = c.fetchall()
    conn.close()

    # Group by session
    sessions = {}
    for session_id, role, user, content, ts in rows:
        sessions.setdefault(session_id, []).append({"role": role, "user": user, "content": content, "ts": ts})

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        total = 0
        for sid, msgs in sessions.items():
            # find user -> assistant pairs in order
            for i in range(len(msgs) - 1):
                m1 = msgs[i]
                m2 = msgs[i + 1]
                if m1["role"] == "user" and m2["role"] == "assistant":
                    prompt = m1["content"]
                    response = m2["content"]
                    # You can include context: join previous N messages
                    j = {"prompt": prompt, "response": response}
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
                    total += 1
        print(f"Wrote {total} examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experiences.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    export_jsonl(args.out)