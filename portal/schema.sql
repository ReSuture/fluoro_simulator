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

-- Session recordings auto-uploaded by devices (see /api/device/videos). The
-- file itself lives at <VIDEO_DIR>/<video_id>.mp4; a video is visible to the
-- emails currently assigned to its device (plus admins), and to anyone in
-- video_shares. NB: reassigning a device therefore hands its library to the
-- new assignee — decommission flows should delete videos first if that matters.
CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,                    -- urlsafe token; filename stem
    device_id   TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    sha256      TEXT NOT NULL,                       -- of the uploaded MP4 (retry dedupe)
    size        INTEGER NOT NULL,                    -- bytes on disk
    duration_s  INTEGER NOT NULL DEFAULT 0,
    recorded_at TEXT,                                -- ISO time from the Pi's file mtime
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (device_id, sha256)
);

-- View-only grants to individual emails (a colleague need not own a device).
CREATE TABLE IF NOT EXISTS video_shares (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    email      TEXT NOT NULL,                        -- stored lowercased
    created_by TEXT NOT NULL,                        -- the sharing owner's email
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (video_id, email)
);
CREATE INDEX IF NOT EXISTS idx_video_shares_email ON video_shares(email);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL DEFAULT (datetime('now')),
    actor     TEXT NOT NULL,       -- admin email or 'provisioner'
    action    TEXT NOT NULL,       -- provision|assign|unassign|delete|cf_error
    device_id TEXT,
    detail    TEXT                 -- JSON blob (email involved, CF ids, error)
);
