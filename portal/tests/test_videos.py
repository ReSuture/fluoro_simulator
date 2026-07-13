"""Device video upload + customer library flows (Cloudflare API mocked)."""

import hashlib
import io
import os
import re

import pytest

from portal import cf_api, config, status

ADMIN = {"X-Dev-Email": "ben@resuture.com"}
CUSTOMER = {"X-Dev-Email": "customer@example.com"}
COLLEAGUE = {"X-Dev-Email": "colleague@example.com"}
STRANGER = {"X-Dev-Email": "stranger@example.com"}
PROVISION_AUTH = {"Authorization": "Bearer test-provision-token"}

MP4 = b"\x00\x00\x00\x18ftypisom-fake-video-bytes" * 100


@pytest.fixture(autouse=True)
def mock_cf(monkeypatch):
    monkeypatch.setattr(cf_api, "create_access_app", lambda hostname, emails: ("app-1", "pol-1"))
    monkeypatch.setattr(cf_api, "create_tunnel", lambda name: "tun-1")
    monkeypatch.setattr(cf_api, "put_tunnel_config", lambda tid, hostname, service=None: None)
    monkeypatch.setattr(cf_api, "create_dns_cname", lambda sub, target: "dns-1")
    monkeypatch.setattr(cf_api, "get_tunnel_token", lambda tid: "fake-tunnel-token")
    monkeypatch.setattr(cf_api, "set_access_policy_emails", lambda app_id, pol_id, emails: None)
    monkeypatch.setattr(cf_api, "delete_access_app", lambda app_id: None)
    monkeypatch.setattr(cf_api, "delete_tunnel", lambda tid: None)
    monkeypatch.setattr(cf_api, "delete_dns_record", lambda rid: None)
    monkeypatch.setattr(status, "get_statuses", lambda: {"tun-1": "healthy"})


def _provision(client, device_id="abc123"):
    resp = client.post("/api/provision", json={"device_id": device_id},
                       headers=PROVISION_AUTH)
    assert resp.status_code == 200
    return resp.get_json()["claim_secret"]


def _csrf(client, headers, page="/library"):
    html = client.get(page, headers=headers).get_data(as_text=True)
    return re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)


def _assign(client, email="customer@example.com", device_id="abc123"):
    client.post("/admin/devices/%s/assign" % device_id,
                data={"email": email, "csrf": _csrf(client, ADMIN, "/admin")},
                headers=ADMIN)


def _upload(client, secret, data=MP4, device_id="abc123", **overrides):
    form = {
        "device_id": device_id,
        "claim_secret": secret,
        "sha256": hashlib.sha256(data).hexdigest(),
        "duration_s": "63",
        "recorded_at": "1783300000",
        "video": (io.BytesIO(data), "fluoro_20260713_190000.mp4"),
    }
    form.update(overrides)
    return client.post("/api/device/videos", data=form,
                       content_type="multipart/form-data")


def _library_video_id(client, headers):
    html = client.get("/library", headers=headers).get_data(as_text=True)
    m = re.search(r"/videos/([A-Za-z0-9_-]+)/stream", html)
    return m.group(1) if m else None


# ── Upload API ────────────────────────────────────────────────────────────────

def test_upload_stores_video(client):
    secret = _provision(client)
    resp = _upload(client, secret)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["duplicate"] is False
    path = os.path.join(config.VIDEO_DIR, data["video_id"] + ".mp4")
    with open(path, "rb") as f:
        assert f.read() == MP4


def test_upload_retry_is_deduped(client):
    secret = _provision(client)
    first = _upload(client, secret).get_json()
    again = _upload(client, secret).get_json()
    assert again["duplicate"] is True
    assert again["video_id"] == first["video_id"]
    assert len(os.listdir(config.VIDEO_DIR)) == 1


def test_upload_rejects_bad_secret(client):
    _provision(client)
    assert _upload(client, "wrong-secret").status_code == 403


def test_upload_rejects_malformed_sha(client):
    secret = _provision(client)
    assert _upload(client, secret, sha256="nothex").status_code == 400


def test_upload_rejects_checksum_mismatch(client):
    secret = _provision(client)
    resp = _upload(client, secret,
                   sha256=hashlib.sha256(b"other bytes").hexdigest())
    assert resp.status_code == 400
    assert not os.listdir(config.VIDEO_DIR)


def test_upload_respects_quota(client, monkeypatch):
    monkeypatch.setattr(config, "VIDEO_QUOTA_BYTES", 10)
    secret = _provision(client)
    assert _upload(client, secret).status_code == 507


# ── Visibility & playback ─────────────────────────────────────────────────────

def test_assigned_customer_sees_and_streams_video(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    html = client.get("/library", headers=CUSTOMER).get_data(as_text=True)
    assert vid in html and html.count("Session ") >= 1
    resp = client.get("/videos/%s/stream" % vid, headers=CUSTOMER)
    assert resp.status_code == 200
    assert resp.data == MP4


def test_stream_supports_range_requests(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    resp = client.get("/videos/%s/stream" % vid,
                      headers=dict(CUSTOMER, Range="bytes=0-99"))
    assert resp.status_code == 206
    assert resp.data == MP4[:100]


def test_stranger_sees_nothing(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    assert vid not in client.get("/library", headers=STRANGER).get_data(as_text=True)
    assert client.get("/videos/%s/stream" % vid, headers=STRANGER).status_code == 403


def test_admin_sees_all_videos(client):
    secret = _provision(client)
    vid = _upload(client, secret).get_json()["video_id"]
    assert vid in client.get("/library", headers=ADMIN).get_data(as_text=True)
    assert client.get("/videos/%s/stream" % vid, headers=ADMIN).status_code == 200


# ── Sharing ───────────────────────────────────────────────────────────────────

def test_share_grants_view_only(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    client.post("/videos/%s/share" % vid,
                data={"email": "colleague@example.com",
                      "csrf": _csrf(client, CUSTOMER)},
                headers=CUSTOMER)
    # The colleague can watch…
    assert client.get("/videos/%s/stream" % vid, headers=COLLEAGUE).status_code == 200
    html = client.get("/library", headers=COLLEAGUE).get_data(as_text=True)
    assert vid in html and "Remove from my library" in html
    # …but not manage.
    assert client.post("/videos/%s/delete" % vid,
                       data={"csrf": _csrf(client, COLLEAGUE)},
                       headers=COLLEAGUE).status_code == 403


def test_shared_user_can_remove_themselves(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    client.post("/videos/%s/share" % vid,
                data={"email": "colleague@example.com",
                      "csrf": _csrf(client, CUSTOMER)},
                headers=CUSTOMER)
    client.post("/videos/%s/unshare" % vid,
                data={"email": "colleague@example.com",
                      "csrf": _csrf(client, COLLEAGUE)},
                headers=COLLEAGUE)
    assert client.get("/videos/%s/stream" % vid, headers=COLLEAGUE).status_code == 403


def test_share_requires_manage(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    resp = client.post("/videos/%s/share" % vid,
                       data={"email": "stranger@example.com",
                             "csrf": _csrf(client, ADMIN, "/admin")},
                       headers=STRANGER)
    assert resp.status_code == 403


# ── Manage ────────────────────────────────────────────────────────────────────

def test_owner_renames_video(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    client.post("/videos/%s/title" % vid,
                data={"title": "Femoral access run 3",
                      "csrf": _csrf(client, CUSTOMER)},
                headers=CUSTOMER)
    assert "Femoral access run 3" in client.get(
        "/library", headers=CUSTOMER).get_data(as_text=True)


def test_owner_deletes_video_and_file(client):
    secret = _provision(client)
    _assign(client)
    vid = _upload(client, secret).get_json()["video_id"]
    client.post("/videos/%s/delete" % vid,
                data={"csrf": _csrf(client, CUSTOMER)},
                headers=CUSTOMER)
    assert client.get("/videos/%s/stream" % vid, headers=CUSTOMER).status_code == 404
    assert not os.listdir(config.VIDEO_DIR)


def test_decommission_removes_video_files(client):
    secret = _provision(client)
    vid = _upload(client, secret).get_json()["video_id"]
    client.post("/admin/devices/abc123/delete",
                data={"csrf": _csrf(client, ADMIN, "/admin")}, headers=ADMIN)
    assert not os.listdir(config.VIDEO_DIR)
    assert client.get("/videos/%s/stream" % vid, headers=ADMIN).status_code == 404
