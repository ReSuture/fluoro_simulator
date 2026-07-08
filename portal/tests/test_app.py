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
    assert resp.status_code == 502
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
    assert resp.status_code == 502
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
