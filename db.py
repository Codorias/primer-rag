import sqlite3
import json
from datetime import datetime

DB_PATH = "chat_history.db"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                sources         TEXT,
                confidence      REAL,
                created_at      TEXT    NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)


def create_conversation(title: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at) VALUES (?, ?)",
            (title[:60], datetime.now().isoformat()),
        )
        return cur.lastrowid


def add_message(conversation_id: int, role: str, content: str,
                sources: list | None = None, confidence: float | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, sources, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                conversation_id, role, content,
                json.dumps(sources) if sources else None,
                confidence,
                datetime.now().isoformat(),
            ),
        )


def get_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def get_messages(conversation_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT role, content, sources, confidence FROM messages
               WHERE conversation_id = ? ORDER BY created_at ASC""",
            (conversation_id,),
        ).fetchall()
    return [
        {
            "role":       r[0],
            "content":    r[1],
            "sources":    json.loads(r[2]) if r[2] else [],
            "confidence": r[3],
        }
        for r in rows
    ]


def delete_conversation(conversation_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages      WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?",              (conversation_id,))


def clear_all_conversations() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
