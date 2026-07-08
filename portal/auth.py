"""Cloudflare Access JWT verification.

Every request that reaches the portal through Cloudflare Access carries a
Cf-Access-Jwt-Assertion header (and a CF_Authorization cookie) — an RS256 JWT
signed by the Zero Trust team's keys. We verify it against the team's JWKS
and require our own application's AUD tag, so a token minted for a different
Access app in the same team cannot open the portal.

Never trust Cf-Access-Authenticated-User-Email alone: it is a plain header
anyone who can reach the origin directly could forge.
"""

import threading
import time
from functools import wraps

import jwt
from jwt import PyJWKClient
from flask import g, request, abort

from . import config

_JWKS_TTL_SECONDS = 24 * 3600

_lock = threading.Lock()
_jwk_client = None
_jwk_client_fetched_at = 0.0


def _jwks_url():
    return "https://%s/cdn-cgi/access/certs" % config.ACCESS_TEAM_DOMAIN


def _get_jwk_client(force_refresh=False):
    """Cached PyJWKClient; recreated daily or when a signing key is unknown
    (Cloudflare rotates Access keys every ~6 weeks)."""
    global _jwk_client, _jwk_client_fetched_at
    with _lock:
        stale = (time.time() - _jwk_client_fetched_at) > _JWKS_TTL_SECONDS
        if _jwk_client is None or stale or force_refresh:
            _jwk_client = PyJWKClient(_jwks_url(), cache_keys=True)
            _jwk_client_fetched_at = time.time()
        return _jwk_client


def _decode(token, signing_key):
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=config.ACCESS_PORTAL_AUD,
        options={"require": ["exp", "aud", "iss"]},
    )
    issuer = "https://%s" % config.ACCESS_TEAM_DOMAIN
    if claims.get("iss") != issuer:
        raise jwt.InvalidIssuerError("unexpected iss %r" % claims.get("iss"))
    return claims


def verify_request():
    """Return the verified, lowercased email for the current request,
    or None if the request carries no valid Access JWT."""
    if config.DEV_MODE:
        dev_email = request.headers.get("X-Dev-Email")
        if dev_email:
            return dev_email.strip().lower()

    token = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get(
        "CF_Authorization"
    )
    if not token:
        return None

    try:
        try:
            signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError:
            # Unknown kid — force one JWKS refresh in case of key rotation.
            signing_key = _get_jwk_client(force_refresh=True).get_signing_key_from_jwt(token)
        claims = _decode(token, signing_key)
    except jwt.PyJWTError:
        return None

    email = claims.get("email", "")
    return email.strip().lower() or None


def require_customer(view):
    """Any Access-authenticated user. Sets g.email and g.is_admin."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        email = verify_request()
        if not email:
            # Access normally intercepts before us; reaching here unauthenticated
            # means a direct origin hit or an expired/foreign token.
            abort(401)
        g.email = email
        g.is_admin = email in config.ADMIN_EMAILS
        return view(*args, **kwargs)

    return wrapper


def require_admin(view):
    """Access-authenticated AND listed in ADMIN_EMAILS."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        email = verify_request()
        if not email:
            abort(401)
        if email not in config.ADMIN_EMAILS:
            abort(403)
        g.email = email
        g.is_admin = True
        return view(*args, **kwargs)

    return wrapper
