"""Provision / assign / unassign flows with the Cloudflare API mocked."""

import re

import pytest

from portal import cf_api, status

ADMIN = {"X-Dev-Email": "ben@resuture.com"}
CUSTOMER = {"X-Dev-Email": "customer@example.com"}
PROVISION_AUTH = {"Authorization": "Bearer test-provision-token"}


@pytest.fixture(autouse=True)
def mock_cf(monkeypatch):
    """Record CF calls; return deterministic fake ids."""
    calls = []

    monkeypatch.setattr(cf_api, "create_access_app",
                        lambda hostname, emails: (calls.append(("access", hostname, sorted(emails))),
                                                  ("app-1", "pol-1"))[1])
    monkeypatch.setattr(cf_api, "create_tunnel",
                        lambda name: (calls.append(("tunnel", name)), "tun-1")[1])
    monkeypatch.setattr(cf_api, "put_tunnel_config",
                        lambda tid, hostname, service=None: calls.append(("config", tid, hostname)))
    monkeypatch.setattr(cf_api, "create_dns_cname",
                        lambda sub, target: (calls.append(("dns", sub, target)), "dns-1")[1])
    monkeypatch.setattr(cf_api, "get_tunnel_token",
                        lambda tid: "fake-tunnel-token-%s" % tid)
    monkeypatch.setattr(cf_api, "set_access_policy_emails",
                        lambda app_id, pol_id, emails: calls.append(("policy", app_id, sorted(emails))))
    monkeypatch.setattr(cf_api, "delete_access_app", lambda app_id: calls.append(("del_access", app_id)))
    monkeypatch.setattr(cf_api, "delete_tunnel", lambda tid: calls.append(("del_tunnel", tid)))
    monkeypatch.setattr(cf_api, "delete_dns_record", lambda rid: calls.append(("del_dns", rid)))
    monkeypatch.setattr(status, "get_statuses", lambda: {"tun-1": "healthy"})
    return calls


def _provision(client, device_id="abc123"):
    return client.post("/api/provision", json={"device_id": device_id},
                       headers=PROVISION_AUTH)


def _csrf(client, page="/admin"):
    html = client.get(page, headers=ADMIN).get_data(as_text=True)
    return re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)


# ── Provisioning ──────────────────────────────────────────────────────────────

def test_provision_creates_device(client, mock_cf):
    resp = _provision(client)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["hostname"] == "sim-abc123.example.com"
    assert data["tunnel_token"] == "fake-tunnel-token-tun-1"
    assert data["existing"] is False
    # Access gate is created BEFORE the DNS record.
    kinds = [c[0] for c in mock_cf]
    assert kinds.index("access") < kinds.index("dns")
    # Fresh device allows admins only.
    assert ("access", "sim-abc123.example.com", ["ben@resuture.com"]) in mock_cf


def test_provision_is_idempotent(client):
    _provision(client)
    resp = _provision(client)
    assert resp.status_code == 200
    assert resp.get_json()["existing"] is True


def test_provision_requires_bearer_token(client):
    resp = client.post("/api/provision", json={"device_id": "abc123"},
                       headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_provision_rejects_bad_device_id(client):
    resp = client.post("/api/provision", json={"device_id": "../etc"},
                       headers=PROVISION_AUTH)
    assert resp.status_code == 400


def test_provision_rolls_back_on_cf_failure(client, mock_cf, monkeypatch):
    monkeypatch.setattr(cf_api, "create_dns_cname",
                        lambda sub, target: (_ for _ in ()).throw(cf_api.CfApiError("dns boom")))
    resp = _provision(client)
    assert resp.status_code == 500
    kinds = [c[0] for c in mock_cf]
    assert "del_tunnel" in kinds and "del_access" in kinds
    # Device must not exist afterwards; re-provisioning starts fresh.
    monkeypatch.setattr(cf_api, "create_dns_cname",
                        lambda sub, target: "dns-1")
    assert _provision(client).get_json()["existing"] is False


# ── Customer view ─────────────────────────────────────────────────────────────

def test_customer_with_no_devices_sees_empty_state(client):
    resp = client.get("/", headers=CUSTOMER)
    assert resp.status_code == 200
    assert b"No simulator is linked" in resp.data


def test_unauthenticated_request_is_rejected(client):
    assert client.get("/").status_code == 401


def test_customer_sees_assigned_device(client):
    _provision(client)
    client.post("/admin/devices/abc123/assign",
                data={"email": "customer@example.com", "csrf": _csrf(client)},
                headers=ADMIN)
    html = client.get("/", headers=CUSTOMER).get_data(as_text=True)
    assert "sim-abc123.example.com" in html
    assert "online" in html


# ── Admin ─────────────────────────────────────────────────────────────────────

def test_admin_page_forbidden_for_customers(client):
    assert client.get("/admin", headers=CUSTOMER).status_code == 403


def test_assign_updates_access_policy(client, mock_cf):
    _provision(client)
    resp = client.post("/admin/devices/abc123/assign",
                       data={"email": "Customer@Example.com", "csrf": _csrf(client)},
                       headers=ADMIN)
    assert resp.status_code == 302
    # Policy now carries customer + admin.
    assert ("policy", "app-1", ["ben@resuture.com", "customer@example.com"]) in mock_cf


def test_assign_rolls_back_if_policy_update_fails(client, monkeypatch):
    _provision(client)
    csrf = _csrf(client)
    monkeypatch.setattr(cf_api, "set_access_policy_emails",
                        lambda *a: (_ for _ in ()).throw(cf_api.CfApiError("policy boom")))
    resp = client.post("/admin/devices/abc123/assign",
                       data={"email": "customer@example.com", "csrf": csrf},
                       headers=ADMIN)
    assert resp.status_code == 500
    # Customer must NOT see the device (portal never claims access CF won't grant).
    html = client.get("/", headers=CUSTOMER).get_data(as_text=True)
    assert "No simulator is linked" in html


def test_assign_requires_csrf(client):
    _provision(client)
    resp = client.post("/admin/devices/abc123/assign",
                       data={"email": "customer@example.com", "csrf": "bogus"},
                       headers=ADMIN)
    assert resp.status_code == 400


def test_unassign_removes_access(client, mock_cf):
    _provision(client)
    csrf = _csrf(client)
    client.post("/admin/devices/abc123/assign",
                data={"email": "customer@example.com", "csrf": csrf}, headers=ADMIN)
    client.post("/admin/devices/abc123/unassign",
                data={"email": "customer@example.com", "csrf": csrf}, headers=ADMIN)
    assert ("policy", "app-1", ["ben@resuture.com"]) in mock_cf
    html = client.get("/", headers=CUSTOMER).get_data(as_text=True)
    assert "No simulator is linked" in html


def test_decommission_deletes_everything(client, mock_cf):
    _provision(client)
    resp = client.post("/admin/devices/abc123/delete",
                       data={"csrf": _csrf(client)}, headers=ADMIN)
    assert resp.status_code == 302
    kinds = [c[0] for c in mock_cf]
    assert {"del_dns", "del_tunnel", "del_access"} <= set(kinds)
    html = client.get("/admin", headers=ADMIN).get_data(as_text=True)
    assert "No devices registered" in html


# ── Device API (claim secret) ─────────────────────────────────────────────────

def _claim(client, secret, email, device_id="abc123"):
    return client.post("/api/device/claim",
                       json={"device_id": device_id, "claim_secret": secret,
                             "email": email})


def _status(client, secret, device_id="abc123"):
    return client.post("/api/device/status",
                       json={"device_id": device_id, "claim_secret": secret})


def test_provision_returns_claim_secret(client, app):
    import hashlib
    data = _provision(client).get_json()
    assert data["claim_secret"]
    with app.app_context():
        from portal import db
        row = db.get_device("abc123")
        assert row["claim_secret_hash"] == \
            hashlib.sha256(data["claim_secret"].encode()).hexdigest()


def test_reprovision_rotates_claim_secret(client):
    old = _provision(client).get_json()["claim_secret"]
    new = _provision(client).get_json()["claim_secret"]
    assert old != new
    assert _status(client, old).status_code == 403
    assert _status(client, new).status_code == 200


def test_claim_assigns_email_and_syncs_policy(client, mock_cf):
    secret = _provision(client).get_json()["claim_secret"]
    resp = _claim(client, secret, "Customer@Example.com")
    assert resp.status_code == 200
    assert resp.get_json()["claimed_email"] == "customer@example.com"
    assert ("policy", "app-1", ["ben@resuture.com", "customer@example.com"]) in mock_cf
    html = client.get("/", headers=CUSTOMER).get_data(as_text=True)
    assert "sim-abc123.example.com" in html


def test_claim_wrong_secret_403(client):
    _provision(client)
    resp = _claim(client, "wrong-secret", "customer@example.com")
    assert resp.status_code == 403
    assert "error" in resp.get_json()
    # Unknown device id gets the same answer (no existence leak).
    resp = _claim(client, "whatever", "customer@example.com", device_id="nosuch")
    assert resp.status_code == 403


def test_claim_bad_email_400(client):
    secret = _provision(client).get_json()["claim_secret"]
    assert _claim(client, secret, "not-an-email").status_code == 400


def test_reclaim_replaces_previous_device_claim(client, mock_cf):
    secret = _provision(client).get_json()["claim_secret"]
    _claim(client, secret, "first@example.com")
    _claim(client, secret, "second@example.com")
    assert mock_cf[-1] == ("policy", "app-1",
                           ["ben@resuture.com", "second@example.com"])
    html = client.get("/", headers={"X-Dev-Email": "first@example.com"}) \
                 .get_data(as_text=True)
    assert "No simulator is linked" in html


def test_claim_preserves_admin_assignments(client, mock_cf):
    secret = _provision(client).get_json()["claim_secret"]
    client.post("/admin/devices/abc123/assign",
                data={"email": "clinician@example.com", "csrf": _csrf(client)},
                headers=ADMIN)
    _claim(client, secret, "first@example.com")
    _claim(client, secret, "second@example.com")
    assert mock_cf[-1] == ("policy", "app-1",
                           ["ben@resuture.com", "clinician@example.com",
                            "second@example.com"])
    html = client.get("/", headers={"X-Dev-Email": "clinician@example.com"}) \
                 .get_data(as_text=True)
    assert "sim-abc123.example.com" in html


def test_claim_rolls_back_on_cf_error(client, monkeypatch):
    secret = _provision(client).get_json()["claim_secret"]
    assert _claim(client, secret, "first@example.com").status_code == 200
    monkeypatch.setattr(cf_api, "set_access_policy_emails",
                        lambda *a: (_ for _ in ()).throw(cf_api.CfApiError("boom")))
    assert _claim(client, secret, "second@example.com").status_code == 500
    # The first claim survives; the failed one left no trace.
    assert _status(client, secret).get_json()["claimed_email"] == "first@example.com"
    html = client.get("/", headers={"X-Dev-Email": "first@example.com"}) \
                 .get_data(as_text=True)
    assert "sim-abc123.example.com" in html


def test_device_status_returns_claimed_email(client):
    secret = _provision(client).get_json()["claim_secret"]
    assert _status(client, secret).get_json()["claimed_email"] is None
    _claim(client, secret, "customer@example.com")
    data = _status(client, secret).get_json()
    assert data["claimed_email"] == "customer@example.com"
    assert data["hostname"] == "sim-abc123.example.com"


def test_device_endpoints_rate_limited(client):
    secret = _provision(client).get_json()["claim_secret"]
    for _ in range(10):
        assert _status(client, secret).status_code == 200
    resp = _status(client, secret)
    assert resp.status_code == 429
    assert "error" in resp.get_json()


def test_init_db_migrates_existing_database(tmp_path, monkeypatch):
    """A DB created before claim_secret_hash existed gains the column in place."""
    import sqlite3
    from portal import config as cfg, db
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE devices (
            device_id        TEXT PRIMARY KEY,
            hostname         TEXT NOT NULL UNIQUE,
            tunnel_id        TEXT NOT NULL UNIQUE,
            access_app_id    TEXT NOT NULL,
            access_policy_id TEXT NOT NULL,
            dns_record_id    TEXT NOT NULL,
            notes            TEXT NOT NULL DEFAULT '',
            created_at       TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.execute("""INSERT INTO devices (device_id, hostname, tunnel_id,
                    access_app_id, access_policy_id, dns_record_id)
                    VALUES ('old123', 'sim-old123.example.com', 't-1', 'a-1',
                            'p-1', 'd-1')""")
    conn.commit()
    conn.close()

    monkeypatch.setattr(cfg, "DATABASE_PATH", path)
    db.init_db()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(devices)")}
    assert "claim_secret_hash" in cols
    row = conn.execute("SELECT * FROM devices").fetchone()
    assert row["device_id"] == "old123"
    assert row["claim_secret_hash"] is None
    conn.close()
