"""Test environment: dev mode + fake Cloudflare settings, set BEFORE the
portal modules are imported so config.py picks them up."""

import os

os.environ.setdefault("PORTAL_DEV", "1")
# A (fake) API token so _sync_access_policy doesn't take its DEV_MODE
# short-circuit — tests assert on the mocked cf_api policy calls.
os.environ.setdefault("CF_API_TOKEN", "test-cf-token")
os.environ.setdefault("ACCESS_TEAM_DOMAIN", "testteam.cloudflareaccess.com")
os.environ.setdefault("ACCESS_PORTAL_AUD", "test-aud-tag")
os.environ.setdefault("ADMIN_EMAILS", "ben@resuture.com")
os.environ.setdefault("PROVISION_TOKEN", "test-provision-token")
os.environ.setdefault("SIM_DOMAIN_SUFFIX", ".example.com")

import pytest

from portal import config


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "portal.db"))
    from portal.app import create_app
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
