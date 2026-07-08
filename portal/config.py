"""Portal configuration.

Everything comes from environment variables (in production, loaded by the
systemd unit's EnvironmentFile=/etc/resuture-portal.env). validate() refuses
to start with missing settings unless dev mode is on, so a misconfigured
deploy fails loudly at boot instead of at the first customer request.
"""

import os


def _env(name, default=None):
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else value


# Dev mode: enables the X-Dev-Email auth bypass (auth.py) and relaxes
# validation so the portal can run on a laptop with no Cloudflare account.
DEV_MODE = _env("PORTAL_DEV", "") == "1"

# Cloudflare API access — the portal is the only holder of this token.
CF_API_TOKEN = _env("CF_API_TOKEN")
CF_ACCOUNT_ID = _env("CF_ACCOUNT_ID")
CF_ZONE_ID = _env("CF_ZONE_ID")  # zone of the device domain (not resuture.com)

# Cloudflare Access (Zero Trust) identity.
ACCESS_TEAM_DOMAIN = _env("ACCESS_TEAM_DOMAIN")  # e.g. resuture.cloudflareaccess.com
ACCESS_PORTAL_AUD = _env("ACCESS_PORTAL_AUD")    # AUD tag of the portal's Access app

# Comma-separated admin emails. Admins see /admin and are kept on every
# device's Access policy so support can always reach a customer's panel.
ADMIN_EMAILS = frozenset(
    e.strip().lower() for e in _env("ADMIN_EMAILS", "").split(",") if e.strip()
)

# Shared secret for POST /api/provision (bench provisioning script).
PROVISION_TOKEN = _env("PROVISION_TOKEN")

DATABASE_PATH = _env("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "portal.db"))

# Device hostname suffix, with leading dot: ".resuturesim.com".
SIM_DOMAIN_SUFFIX = _env("SIM_DOMAIN_SUFFIX", "")

# Flask session signing (only used for the admin CSRF token cookie).
FLASK_SECRET_KEY = _env("FLASK_SECRET_KEY")

# What each device's tunnel forwards to on the Pi.
ORIGIN_SERVICE = "http://localhost:5000"

_REQUIRED = [
    "CF_API_TOKEN", "CF_ACCOUNT_ID", "CF_ZONE_ID",
    "ACCESS_TEAM_DOMAIN", "ACCESS_PORTAL_AUD",
    "PROVISION_TOKEN", "SIM_DOMAIN_SUFFIX", "FLASK_SECRET_KEY",
]


def validate():
    """Raise RuntimeError listing every missing required setting."""
    if DEV_MODE:
        return
    missing = [name for name in _REQUIRED if not globals()[name]]
    if not ADMIN_EMAILS:
        missing.append("ADMIN_EMAILS")
    if SIM_DOMAIN_SUFFIX and not SIM_DOMAIN_SUFFIX.startswith("."):
        raise RuntimeError("SIM_DOMAIN_SUFFIX must start with '.' (e.g. .resuturesim.com)")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: %s "
            "(set them in /etc/resuture-portal.env, or PORTAL_DEV=1 for local dev)"
            % ", ".join(sorted(missing))
        )


def device_hostname(device_id):
    return "sim-%s%s" % (device_id, SIM_DOMAIN_SUFFIX)
