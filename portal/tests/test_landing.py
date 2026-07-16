"""The public landing page on the device domain's apex/www hosts.

conftest sets SIM_DOMAIN_SUFFIX=.example.com, so the landing hosts are
example.com and www.example.com; every other host keeps portal behaviour.
"""


def test_apex_serves_landing(client):
    r = client.get("/", headers={"Host": "example.com"})
    assert r.status_code == 200
    assert b"ReSuture" in r.data
    assert b"portal.example.com" in r.data


def test_www_serves_landing(client):
    r = client.get("/", headers={"Host": "www.example.com"})
    assert r.status_code == 200
    assert b"NAVISLab" in r.data


def test_landing_needs_no_auth(client):
    # No Access JWT, no X-Dev-Email — the landing page must still render.
    r = client.get("/", headers={"Host": "example.com"})
    assert r.status_code == 200


def test_other_paths_on_apex_redirect_home(client):
    for path in ("/admin", "/library", "/api/my/devices", "/healthz"):
        r = client.get(path, headers={"Host": "example.com"})
        assert r.status_code == 302, path
        assert r.headers["Location"].endswith("/")


def test_static_assets_load_on_landing_hosts(client):
    # The hero logo must not be caught by the redirect-home rule.
    r = client.get("/static/logosign_white.png", headers={"Host": "example.com"})
    assert r.status_code == 200


def test_apex_post_redirects(client):
    r = client.post("/api/device/status", headers={"Host": "www.example.com"})
    assert r.status_code == 302


def test_portal_host_unaffected(client):
    # The usual host still runs the portal: unauthenticated / is a 401.
    r = client.get("/")
    assert r.status_code == 401
