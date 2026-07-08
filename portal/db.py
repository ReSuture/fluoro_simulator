"""SQLite helpers for the portal.

One connection per request (Flask g), WAL mode so the status poller and a
writing admin request don't block each other. SQLite is plenty at this scale:
tens to hundreds of devices, a single writer process.
"""

import json
import os
import sqlite3

from flask import g

from . import config

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def _connect():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create the database file and schema if needed. Safe to run every boot."""
    conn = _connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        with open(_SCHEMA_PATH, encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


def get_db():
    if "db" not in g:
        g.db = _connect()
    return g.db


def close_db(_exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


# ── Queries ───────────────────────────────────────────────────────────────────

def devices_for_email(email):
    return get_db().execute(
        """SELECT d.* FROM devices d
           JOIN assignments a ON a.device_id = d.device_id
           WHERE a.email = ? ORDER BY d.created_at""",
        (email.lower(),),
    ).fetchall()


def all_devices():
    return get_db().execute("SELECT * FROM devices ORDER BY created_at").fetchall()


def get_device(device_id):
    return get_db().execute(
        "SELECT * FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()


def device_emails(device_id):
    rows = get_db().execute(
        "SELECT email FROM assignments WHERE device_id = ? ORDER BY email",
        (device_id,),
    ).fetchall()
    return [r["email"] for r in rows]


def insert_device(device_id, hostname, tunnel_id, access_app_id,
                  access_policy_id, dns_record_id, notes=""):
    conn = get_db()
    conn.execute(
        """INSERT INTO devices (device_id, hostname, tunnel_id, access_app_id,
                                access_policy_id, dns_record_id, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (device_id, hostname, tunnel_id, access_app_id,
         access_policy_id, dns_record_id, notes),
    )
    conn.commit()


def add_assignment(device_id, email, created_by):
    conn = get_db()
    conn.execute(
        """INSERT OR IGNORE INTO assignments (device_id, email, created_by)
           VALUES (?, ?, ?)""",
        (device_id, email.lower(), created_by),
    )
    conn.commit()


def remove_assignment(device_id, email):
    conn = get_db()
    conn.execute(
        "DELETE FROM assignments WHERE device_id = ? AND email = ?",
        (device_id, email.lower()),
    )
    conn.commit()


def delete_device(device_id):
    conn = get_db()
    conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
    conn.commit()


def audit(actor, action, device_id=None, **detail):
    conn = get_db()
    conn.execute(
        "INSERT INTO audit_log (actor, action, device_id, detail) VALUES (?, ?, ?, ?)",
        (actor, action, device_id, json.dumps(detail) if detail else None),
    )
    conn.commit()
