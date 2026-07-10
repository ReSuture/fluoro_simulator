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
    # Generous busy timeout: if two processes ever touch the DB at once
    # (e.g. workers booting together), wait for the lock instead of erroring.
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=30)
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
        # schema.sql is CREATE TABLE IF NOT EXISTS, so databases created before
        # a column existed never gain it; patch such columns in place here.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(devices)")}
        if "claim_secret_hash" not in cols:
            conn.execute("ALTER TABLE devices ADD COLUMN claim_secret_hash TEXT")
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
                  access_policy_id, dns_record_id, notes="",
                  claim_secret_hash=None):
    conn = get_db()
    conn.execute(
        """INSERT INTO devices (device_id, hostname, tunnel_id, access_app_id,
                                access_policy_id, dns_record_id, notes,
                                claim_secret_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (device_id, hostname, tunnel_id, access_app_id,
         access_policy_id, dns_record_id, notes, claim_secret_hash),
    )
    conn.commit()


def set_claim_secret_hash(device_id, claim_secret_hash):
    conn = get_db()
    conn.execute(
        "UPDATE devices SET claim_secret_hash = ? WHERE device_id = ?",
        (claim_secret_hash, device_id),
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


def get_assignments(device_id):
    """All assignment rows for a device as (email, created_by) tuples."""
    rows = get_db().execute(
        """SELECT email, created_by FROM assignments
           WHERE device_id = ? ORDER BY email""",
        (device_id,),
    ).fetchall()
    return [(r["email"], r["created_by"]) for r in rows]


def replace_device_claim(device_id, email):
    """Make `email` the device's one device-claimed assignment.

    Atomically removes any prior created_by='device' rows plus any existing
    row for the same email (so an email an admin already added becomes the
    device claim deterministically), then inserts the new claim. Returns the
    removed (email, created_by) rows so a failed Cloudflare sync can restore
    them via restore_assignments().
    """
    email = email.lower()
    conn = get_db()
    removed = conn.execute(
        """SELECT email, created_by FROM assignments
           WHERE device_id = ? AND (created_by = 'device' OR email = ?)""",
        (device_id, email),
    ).fetchall()
    removed = [(r["email"], r["created_by"]) for r in removed]
    conn.execute(
        """DELETE FROM assignments
           WHERE device_id = ? AND (created_by = 'device' OR email = ?)""",
        (device_id, email),
    )
    conn.execute(
        "INSERT INTO assignments (device_id, email, created_by) VALUES (?, ?, 'device')",
        (device_id, email),
    )
    conn.commit()
    return removed


def restore_assignments(device_id, rows):
    """Re-insert assignment rows removed by replace_device_claim (rollback)."""
    conn = get_db()
    conn.executemany(
        """INSERT OR IGNORE INTO assignments (device_id, email, created_by)
           VALUES (?, ?, ?)""",
        [(device_id, email, created_by) for (email, created_by) in rows],
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
