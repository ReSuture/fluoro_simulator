-- FluoroSim portal schema. Applied idempotently at startup (db.init_db).

CREATE TABLE IF NOT EXISTS devices (
    device_id        TEXT PRIMARY KEY,              -- e.g. 'a1b2c3'
    hostname         TEXT NOT NULL UNIQUE,          -- 'sim-a1b2c3.<device-domain>'
    tunnel_id        TEXT NOT NULL UNIQUE,          -- Cloudflare tunnel UUID
    access_app_id    TEXT NOT NULL,                 -- Access application UUID
    access_policy_id TEXT NOT NULL,                 -- the allow policy edited on assign
    dns_record_id    TEXT NOT NULL,                 -- for clean decommission
    notes            TEXT NOT NULL DEFAULT '',
    claim_secret_hash TEXT,                         -- sha256 hex of the device's claim secret
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- A device can serve several emails (a training center's clinicians) and one
-- email can own several devices; both fall out of this join table.
CREATE TABLE IF NOT EXISTS assignments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id  TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    email      TEXT NOT NULL,                       -- stored lowercased
    created_by TEXT NOT NULL,                       -- admin email
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (device_id, email)
);
CREATE INDEX IF NOT EXISTS idx_assignments_email ON assignments(email);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL DEFAULT (datetime('now')),
    actor     TEXT NOT NULL,       -- admin email or 'provisioner'
    action    TEXT NOT NULL,       -- provision|assign|unassign|delete|cf_error
    device_id TEXT,
    detail    TEXT                 -- JSON blob (email involved, CF ids, error)
);
