"""Cloudflare API client — the only place the CF API token is used.

Thin wrappers over https://api.cloudflare.com/client/v4 covering exactly what
the portal needs: named tunnels (create / token / ingress config / status /
delete), proxied DNS CNAMEs, and per-hostname Access applications with an
email-allowlist policy.

Token scopes (least privilege):
  - Zone / DNS / Edit               (device-domain zone only)
  - Account / Access: Apps and Policies / Edit
  - Account / Cloudflare Tunnel / Edit
"""

import time

import requests

from . import config

_BASE = "https://api.cloudflare.com/client/v4"
_TIMEOUT = 20


class CfApiError(RuntimeError):
    def __init__(self, message, errors=None, status=None):
        super().__init__(message)
        self.errors = errors or []
        self.status = status


def _request(method, path, **kwargs):
    url = _BASE + path
    headers = {"Authorization": "Bearer %s" % config.CF_API_TOKEN}
    for attempt in (1, 2):
        resp = requests.request(method, url, headers=headers, timeout=_TIMEOUT, **kwargs)
        # One retry on rate-limit or transient server error.
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == 1:
                time.sleep(2)
                continue
        break
    try:
        body = resp.json()
    except ValueError:
        raise CfApiError(
            "Cloudflare API returned non-JSON (HTTP %d) for %s %s"
            % (resp.status_code, method, path),
            status=resp.status_code,
        )
    if not body.get("success", False):
        raise CfApiError(
            "Cloudflare API error for %s %s: %s" % (method, path, body.get("errors")),
            errors=body.get("errors"),
            status=resp.status_code,
        )
    return body["result"]


def _acct(path):
    return "/accounts/%s%s" % (config.CF_ACCOUNT_ID, path)


def _zone(path):
    return "/zones/%s%s" % (config.CF_ZONE_ID, path)


# ── Tunnels ───────────────────────────────────────────────────────────────────

def create_tunnel(name):
    """Create a remotely-managed tunnel; returns its UUID."""
    result = _request("POST", _acct("/cfd_tunnel"),
                      json={"name": name, "config_src": "cloudflare"})
    return result["id"]


def get_tunnel_token(tunnel_id):
    """The single-tunnel credential the Pi runs cloudflared with."""
    return _request("GET", _acct("/cfd_tunnel/%s/token" % tunnel_id))


def put_tunnel_config(tunnel_id, hostname, service=None):
    """Ingress: hostname -> the Pi-local panel; everything else 404s."""
    service = service or config.ORIGIN_SERVICE
    _request("PUT", _acct("/cfd_tunnel/%s/configurations" % tunnel_id), json={
        "config": {
            "ingress": [
                {"hostname": hostname, "service": service},
                {"service": "http_status:404"},
            ]
        }
    })


def list_tunnel_statuses():
    """{tunnel_id: status} for the whole fleet in one paginated listing.
    Status values: healthy | degraded | down | inactive."""
    statuses = {}
    page = 1
    while True:
        result = _request(
            "GET", _acct("/cfd_tunnel"),
            params={"is_deleted": "false", "per_page": 100, "page": page},
        )
        for tunnel in result:
            statuses[tunnel["id"]] = tunnel.get("status", "inactive")
        if len(result) < 100:
            return statuses
        page += 1


def delete_tunnel(tunnel_id):
    # Drop any live connections first, otherwise the delete is rejected.
    try:
        _request("DELETE", _acct("/cfd_tunnel/%s/connections" % tunnel_id))
    except CfApiError:
        pass  # no connections to drop
    _request("DELETE", _acct("/cfd_tunnel/%s" % tunnel_id))


# ── DNS ───────────────────────────────────────────────────────────────────────

def create_dns_cname(subdomain, target):
    """Proxied CNAME (orange cloud — must stay proxied so Access applies).
    Returns the record id."""
    result = _request("POST", _zone("/dns_records"), json={
        "type": "CNAME",
        "name": subdomain,
        "content": target,
        "proxied": True,
    })
    return result["id"]


def delete_dns_record(record_id):
    _request("DELETE", _zone("/dns_records/%s" % record_id))


# ── Access applications & policies ────────────────────────────────────────────

def _email_includes(emails):
    return [{"email": {"email": e}} for e in sorted(set(emails))]


def create_access_app(hostname, allowed_emails):
    """Self-hosted Access app protecting `hostname`, with one allow policy.
    Returns (app_id, policy_id)."""
    app = _request("POST", _acct("/access/apps"), json={
        "name": "FluoroSim %s" % hostname,
        "domain": hostname,
        "type": "self_hosted",
        "session_duration": "24h",
        "app_launcher_visible": False,
    })
    policy = _request("POST", _acct("/access/apps/%s/policies" % app["id"]), json={
        "name": "allowed users",
        "decision": "allow",
        "include": _email_includes(allowed_emails),
    })
    return app["id"], policy["id"]


def set_access_policy_emails(app_id, policy_id, emails):
    _request("PUT", _acct("/access/apps/%s/policies/%s" % (app_id, policy_id)), json={
        "name": "allowed users",
        "decision": "allow",
        "include": _email_includes(emails),
    })


def delete_access_app(app_id):
    _request("DELETE", _acct("/access/apps/%s" % app_id))
