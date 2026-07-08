"""JWT verification against a locally generated RS256 keypair — no network."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask

from portal import auth, config


@pytest.fixture(scope="module")
def keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


@pytest.fixture()
def flask_ctx():
    return Flask(__name__)


class _StubSigningKey:
    def __init__(self, public_key):
        self.key = public_key


class _StubJwkClient:
    def __init__(self, public_key):
        self._key = _StubSigningKey(public_key)

    def get_signing_key_from_jwt(self, _token):
        return self._key


@pytest.fixture(autouse=True)
def stub_jwks(monkeypatch, keypair):
    _, public = keypair
    monkeypatch.setattr(auth, "_get_jwk_client",
                        lambda force_refresh=False: _StubJwkClient(public))


def _token(private, **overrides):
    claims = {
        "aud": config.ACCESS_PORTAL_AUD,
        "iss": "https://%s" % config.ACCESS_TEAM_DOMAIN,
        "exp": int(time.time()) + 600,
        "email": "Customer@Example.com",
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test"})


def _verify_with(flask_ctx, token):
    headers = {"Cf-Access-Jwt-Assertion": token} if token else {}
    with flask_ctx.test_request_context(headers=headers):
        return auth.verify_request()


def test_valid_token_returns_lowercased_email(flask_ctx, keypair):
    private, _ = keypair
    assert _verify_with(flask_ctx, _token(private)) == "customer@example.com"


def test_wrong_audience_rejected(flask_ctx, keypair):
    private, _ = keypair
    assert _verify_with(flask_ctx, _token(private, aud="other-app-aud")) is None


def test_wrong_issuer_rejected(flask_ctx, keypair):
    private, _ = keypair
    bad = _token(private, iss="https://evil.cloudflareaccess.com")
    assert _verify_with(flask_ctx, bad) is None


def test_expired_token_rejected(flask_ctx, keypair):
    private, _ = keypair
    assert _verify_with(flask_ctx, _token(private, exp=int(time.time()) - 10)) is None


def test_missing_token_rejected(flask_ctx):
    assert _verify_with(flask_ctx, None) is None


def test_wrong_key_rejected(flask_ctx, keypair):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert _verify_with(flask_ctx, _token(other)) is None


def test_dev_header_bypass_only_in_dev_mode(flask_ctx, monkeypatch):
    with flask_ctx.test_request_context(headers={"X-Dev-Email": "Dev@Example.com"}):
        assert auth.verify_request() == "dev@example.com"
    monkeypatch.setattr(config, "DEV_MODE", False)
    with flask_ctx.test_request_context(headers={"X-Dev-Email": "dev@example.com"}):
        assert auth.verify_request() is None
