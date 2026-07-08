"""Fleet online-status, from Cloudflare's tunnel status.

"Online" means the device's cloudflared is connected to Cloudflare's edge —
the same condition that makes the panel reachable. (It does not prove the
FluoroSim process itself is healthy, but systemd restarts that on crash.)

One API listing covers the whole fleet, cached briefly so page polling never
hammers the Cloudflare API. gunicorn runs a couple of workers, so worst case
there are that many independent caches — fine.
"""

import threading
import time

from . import cf_api, config

_CACHE_TTL_SECONDS = 30

_lock = threading.Lock()
_cache = {}
_cache_at = 0.0


def get_statuses():
    """{tunnel_id: 'healthy'|'degraded'|'down'|'inactive'} — cached ~30s.
    On API failure, serves the last known statuses rather than erroring
    the customer page."""
    global _cache, _cache_at
    with _lock:
        if time.time() - _cache_at < _CACHE_TTL_SECONDS:
            return dict(_cache)
    if config.DEV_MODE and not config.CF_API_TOKEN:
        return {}
    try:
        fresh = cf_api.list_tunnel_statuses()
    except Exception:
        with _lock:
            return dict(_cache)
    with _lock:
        _cache = fresh
        _cache_at = time.time()
        return dict(_cache)


def is_online(status):
    return status in ("healthy", "degraded")
