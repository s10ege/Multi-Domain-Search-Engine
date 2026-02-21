# auth_store.py
# User management, password hashing, and auth event logging.

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional


# ---------- Password hashing (PBKDF2-HMAC-SHA256, stdlib only) ----------

def _hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}:{key.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":", 1)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 100_000)
        return key.hex() == key_hex
    except Exception:
        return False


# ---------- DB init ----------

def init_auth_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT,
                provider      TEXT    NOT NULL DEFAULT 'local',
                github_id     TEXT,
                created_at    TEXT    NOT NULL,
                last_login    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER,
                username   TEXT,
                event      TEXT NOT NULL,
                provider   TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


# ---------- User creation ----------

def create_user(db_path: str, username: str, email: str, password: str) -> Optional[int]:
    """Create a local (username/password) user. Returns new id, or None on duplicate."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash, provider, created_at) "
                "VALUES (?, ?, ?, 'local', ?)",
                (username.strip(), email.strip().lower(), _hash_password(password), now),
            )
            conn.commit()
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def create_github_user(
    db_path: str, email: str, display_name: str, github_id: str
) -> Optional[int]:
    """Create a GitHub-provider user. Returns new id, or None if email already taken."""
    now = datetime.now(timezone.utc).isoformat()
    base = (display_name or email.split("@")[0]).strip().replace(" ", "_") or "user"
    username = base
    try:
        with sqlite3.connect(db_path) as conn:
            suffix = 1
            while conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone():
                username = f"{base}_{suffix}"
                suffix += 1
            cur = conn.execute(
                "INSERT INTO users (username, email, provider, github_id, created_at) "
                "VALUES (?, ?, 'github', ?, ?)",
                (username, email.strip().lower(), str(github_id), now),
            )
            conn.commit()
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


# ---------- Lookups ----------

def get_user_by_email(db_path: str, email: str) -> Optional[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_username(db_path: str, username: str) -> Optional[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
        ).fetchone()
    return dict(row) if row else None


# ---------- Updates ----------

def link_github(db_path: str, user_id: int, github_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET github_id = ? WHERE id = ?", (str(github_id), user_id)
        )
        conn.commit()


def update_last_login(db_path: str, user_id: int) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), user_id),
        )
        conn.commit()


# ---------- Auth log ----------

def log_event(
    db_path: str,
    user_id: Optional[int],
    username: Optional[str],
    event: str,
    provider: str = "local",
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO auth_log (user_id, username, event, provider, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, event, provider, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


# ---------- Session helper ----------

def build_session_user(user: dict, admin_emails: list[str]) -> dict:
    """Build the dict stored in st.session_state.auth_user."""
    email = (user.get("email") or "").lower()
    return {
        "id": user["id"],
        "username": user.get("username") or email.split("@")[0],
        "email": email,
        "provider": user.get("provider", "local"),
        "is_admin": email in [e.lower() for e in admin_emails],
    }


# ---------- Admin queries ----------

def list_all_users(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, username, email, provider, github_id, created_at, last_login "
            "FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def list_auth_log(db_path: str, limit: int = 200) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, user_id, username, event, provider, created_at "
            "FROM auth_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
