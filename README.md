# AI_Stream — interactive AI for streams

Purpose
- A minimal scaffold to build an AI that learns from conversations ("experience") while interacting with viewers.
- Start collecting conversational data, export it as training data, and run periodic fine-tuning / updates.

Goals
- Provide a running chat server for streaming use (WebSocket + REST).
- Log every conversation (user ↔ assistant) as your experience store.
- Export experiences as JSONL for periodic training (rehearsal/experience replay).
- Provide a trainer skeleton using Hugging Face / PEFT for periodic updates.

Quick architecture
- FastAPI server (app.py) — accepts chat messages, replies (placeholder model), logs messages.
- SQLite (conversations.db) — persistent store of messages and sessions.
- export_experiences.py — aggregate sessions into JSONL training examples.
- trainer.py — skeleton for fine-tuning with Hugging Face / PEFT (LoRA) or offline training.
- Iteration pattern: collect conversations → export → train on combined (past + new) data → deploy updated weights → repeat.

First 30 minutes (hands-on)
1) Clone or create repo and create virtualenv:
   python -m venv venv
   source venv/bin/activate   (Windows: venv\Scripts\activate)

2) Install dependencies:
   pip install -r requirements.txt

3) Start the server:
   uvicorn app:app --reload --port 8000

4) Try the REST chat:
   POST http://localhost:8000/chat
   Body: {"session_id":"test1","user":"viewer1","message":"Hello AI!"}

   Or use WebSocket at ws://localhost:8000/ws and send JSON messages:
   {"session_id":"test1","user":"viewer1","message":"Hey, how are you?"}

5) Inspect SQLite: conversations.db. Export experiences:
   python scripts/export_experiences.py --out data/experiences.jsonl

6) After collecting some examples, run trainer skeleton:
   python trainer.py --data data/experiences.jsonl --output-dir models/latest

How learning works (recommended safe approach)
- Do NOT continuously update model weights after every single message (unstable, catastrophic forgetting).
- Use experience replay / rehearsal:
  - Continuously log interactions.
  - Periodically (daily / weekly) aggregate new experiences and sample a training set that mixes archived examples with recent ones.
  - Fine-tune a model using that mixed dataset (centered on desired behavior).
  - Validate on a held-out test set before deploying.
- Optionally use RL from Human Feedback (RLHF) later to tune preferences, or use supervised fine-tuning + preference ranking.

Data format (exported JSONL)
- Each line is {"prompt": "<conversation so far + user message>", "response": "<assistant reply>"}
- Trainer will use these as supervised examples.

Safety & privacy
- Be careful with PII: filter or redact names, credit cards, personal info from logs before training.
- Rate limit public endpoints and add moderation/filters for toxic outputs.
- Keep logs encrypted or on private storage if they contain sensitive info.

Next milestones
- Short term (1–2 weeks): collect sample conversations (do test streams), run basic fine-tune and redeploy.
- Mid term (1–2 months): add embeddings + RAG (vector DB) for knowledge, add memory across sessions.
- Longer term: implement human-in-the-loop ranking and RLHF to shape behavior.

If you want, I can:
- Push these files as a commit to your GitHub repo (I can scaffold PR contents).
- Replace the placeholder reply with an integration to OpenAI or a local HF model.
- Add a Vector DB (Weaviate/FAISS/Chroma) and RAG pipeline for long-term memory.

Read below for the actual files (server, exporter, trainer).
