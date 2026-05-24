"""
governance_logger.py - Fire-and-forget telemetry for agent events.

Three event types:
  - user_message  : a user prompt
  - tool_call     : an agent invoked a tool
  - bash_call     : an agent ran a shell command

Each event is:
  1. Sanitised (API keys, OAuth tokens, JWTs, home-dir paths scrubbed).
  2. Sent to the routing endpoint (server.py /route) to get a verdict
     (category, top skill, envelope state, projected x/y/z).
  3. Appended to a local SQLite log with WAL mode enabled.

All three steps are async and never block the caller. If the routing
endpoint is down, the event is still logged with envelope_state="unknown".

This is the production sister of the demo viewer. The viewer is for
humans to look at; this module is for agents to feed.
"""

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

POINTCLOUD_URL = os.environ.get("POINTCLOUD_URL", "http://localhost:8300")
CORPUS_SLUG = os.environ.get("ROUTE_CORPUS_SLUG", "skills")
ROUTE_ENDPOINT = f"{POINTCLOUD_URL}/api/corpus/{CORPUS_SLUG}/route-multihop"
DB_PATH = Path(os.environ.get("GOVERNANCE_DB", "./data/governance.db"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "10"))

# --- Sanitisation ---------------------------------------------------------

# Secrets we never want to ship to a remote service or write to disk.
_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),       "<openai-key>"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),   "<anthropic-key>"),
    (re.compile(r"ya29\.[A-Za-z0-9_\-]+"),        "<oauth-token>"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
                                                  "<jwt>"),
    (re.compile(r"://[^:/\s]+:[^@/\s]+@"),        "://<user>:<pass>@"),
    (re.compile(r"AKIA[0-9A-Z]{16}"),             "<aws-key>"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"),          "<github-token>"),
]

# Collapse absolute home paths so the log is portable across machines.
_HOME = str(Path.home())


def sanitise(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    out = text.replace(_HOME, "~")
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


# --- Storage --------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    text TEXT NOT NULL,
    envelope_state TEXT,
    category TEXT,
    top_skill TEXT,
    top_skill_snippet TEXT,
    max_sim REAL,
    x REAL, y REAL, z REAL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    tool_input TEXT,
    envelope_state TEXT,
    category TEXT,
    top_skill TEXT,
    top_skill_snippet TEXT,
    max_sim REAL,
    x REAL, y REAL, z REAL
);
CREATE TABLE IF NOT EXISTS bash_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    command TEXT NOT NULL,
    envelope_state TEXT,
    category TEXT,
    top_skill TEXT,
    top_skill_snippet TEXT,
    max_sim REAL,
    x REAL, y REAL, z REAL
);
"""


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)


_init_db()


# --- Routing --------------------------------------------------------------

async def _route(text: str) -> dict[str, Any]:
    """
    Ask server.py to place this text in the corpus. Returns a dict with
    envelope_state, category, top_skill, top_skill_snippet, max_sim,
    and x/y/z coordinates. On any error, returns a stub with envelope_state
    set to 'unknown' so the log row is still useful.
    """
    payload = {"text": sanitise(text)[:4000]}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as cx:
            r = await cx.post(ROUTE_ENDPOINT, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return {"envelope_state": "unknown", "category": None,
                "top_skill": None, "top_skill_snippet": None,
                "max_sim": None, "x": None, "y": None, "z": None}

    cands = data.get("candidates") or []
    if not cands:
        return {"envelope_state": "out", "category": None,
                "top_skill": None, "top_skill_snippet": None,
                "max_sim": None, "x": None, "y": None, "z": None}

    top = cands[0]
    sim = float(top.get("sim", 0.0))
    if sim >= 0.45:
        env = "in"
    elif sim >= 0.30:
        env = "edge"
    else:
        env = "out"

    return {
        "envelope_state": env,
        "category": top.get("category"),
        "top_skill": top.get("label"),
        "top_skill_snippet": top.get("snippet"),
        "max_sim": sim,
        "x": data.get("query_xyz", [None, None, None])[0],
        "y": data.get("query_xyz", [None, None, None])[1],
        "z": data.get("query_xyz", [None, None, None])[2],
    }


# --- Insert helpers -------------------------------------------------------

def _insert(table: str, cols: list[str], values: list[Any]) -> None:
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(sql, values)


# --- Public API -----------------------------------------------------------

async def log_user_message(text: str, session_id: str | None = None) -> None:
    verdict = await _route(text)
    _insert(
        "user_messages",
        ["ts", "session_id", "text",
         "envelope_state", "category", "top_skill", "top_skill_snippet",
         "max_sim", "x", "y", "z"],
        [time.time(), session_id, sanitise(text),
         verdict["envelope_state"], verdict["category"],
         verdict["top_skill"], verdict["top_skill_snippet"],
         verdict["max_sim"], verdict["x"], verdict["y"], verdict["z"]],
    )


async def log_tool_call(tool_name: str, tool_input: dict | str,
                        session_id: str | None = None) -> None:
    text = (f"{tool_name}: " +
            (json.dumps(tool_input) if isinstance(tool_input, dict)
             else str(tool_input)))
    verdict = await _route(text)
    _insert(
        "tool_calls",
        ["ts", "session_id", "tool_name", "tool_input",
         "envelope_state", "category", "top_skill", "top_skill_snippet",
         "max_sim", "x", "y", "z"],
        [time.time(), session_id, tool_name, sanitise(text),
         verdict["envelope_state"], verdict["category"],
         verdict["top_skill"], verdict["top_skill_snippet"],
         verdict["max_sim"], verdict["x"], verdict["y"], verdict["z"]],
    )


async def log_bash_call(command: str, session_id: str | None = None) -> None:
    verdict = await _route(command)
    _insert(
        "bash_calls",
        ["ts", "session_id", "command",
         "envelope_state", "category", "top_skill", "top_skill_snippet",
         "max_sim", "x", "y", "z"],
        [time.time(), session_id, sanitise(command),
         verdict["envelope_state"], verdict["category"],
         verdict["top_skill"], verdict["top_skill_snippet"],
         verdict["max_sim"], verdict["x"], verdict["y"], verdict["z"]],
    )


def fire_and_forget(coro) -> None:
    """
    Schedule a logging coroutine without awaiting it. Use this from
    sync code paths inside your agent. Never raises.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(coro)
        else:
            loop.run_until_complete(coro)
    except Exception:
        pass
