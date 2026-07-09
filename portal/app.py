"""ReSuture FluoroSim portal.

Customers sign in through Cloudflare Access and land on a page listing their
simulator(s), each linking to https://sim-<device-id>.<device-domain>/ — the
FluoroSim panel on their own Raspberry Pi, gated by its own Access policy.

Admins additionally get /admin: register-on-provision device inventory,
assign/unassign customers (which edits the device's Cloudflare Access policy),
and decommission.

Run (production):  gunicorn -w 2 -b 127.0.0.1:8080 'portal.app:create_app()'
Run (local dev):   PORTAL_DEV=1 flask --app portal.app run
                   then send an X-Dev-Email header to impersonate users.
"""

import hmac
import re
import secrets

from flask import (Flask, abort, g, jsonify, redirect, render_template,
                   request, session, url_for)

from . import cf_api, config, db, status
from .auth import require_admin, require_customer

_DEVICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app():
    config.validate()
    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY or secrets.token_hex(32)
    db.init_db()
    app.teardown_appcontext(db.close_db)

    # ── CSRF (admin forms) ────────────────────────────────────────────────────

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    def check_csrf():
        sent = request.form.get("csrf", "")
        if not hmac.compare_digest(sent, session.get("csrf", "-")):
            abort(400, description="Bad or missing CSRF token — reload the page.")

    # ── Customer pages ────────────────────────────────────────────────────────

    def _device_cards(email):
        statuses = status.get_statuses()
        return [
            {
                "device_id": d["device_id"],
                "hostname": d["hostname"],
                "url": "https://%s/" % d["hostname"],
                "online": status.is_online(statuses.get(d["tunnel_id"])),
                "status": statuses.get(d["tunnel_id"], "unknown"),
            }
            for d in db.devices_for_email(email)
        ]

    @app.get("/")
    @require_customer
    def home():
        return render_template("devices.html", devices=_device_cards(g.email),
                               email=g.email, is_admin=g.is_admin)

    @app.get("/api/my/devices")
    @require_customer
    def my_devices():
        return jsonify(devices=_device_cards(g.email))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ── Admin ─────────────────────────────────────────────────────────────────

    def _admin_rows():
        statuses = status.get_statuses()
        rows = []
        for d in db.all_devices():
            rows.append({
                "device_id": d["device_id"],
                "hostname": d["hostname"],
                "notes": d["notes"],
                "created_at": d["created_at"],
                "emails": db.device_emails(d["device_id"]),
                "status": statuses.get(d["tunnel_id"], "unknown"),
                "online": status.is_online(statuses.get(d["tunnel_id"])),
            })
        return rows

    @app.get("/admin")
    @require_admin
    def admin():
        return render_template("admin.html", devices=_admin_rows(), email=g.email)

    def _sync_access_policy(device):
        """Push the device's allowed-email set (customers + admins) to its
        Cloudflare Access policy. Admins stay on every policy for support."""
        emails = set(db.device_emails(device["device_id"])) | set(config.ADMIN_EMAILS)
        cf_api.set_access_policy_emails(
            device["access_app_id"], device["access_policy_id"], emails
        )

    @app.post("/admin/devices/<device_id>/assign")
    @require_admin
    def assign(device_id):
        check_csrf()
        device = db.get_device(device_id) or abort(404)
        email = request.form.get("email", "").strip().lower()
        if not _EMAIL_RE.match(email):
            abort(400, description="That doesn't look like an email address.")
        db.add_assignment(device_id, email, g.email)
        try:
            _sync_access_policy(device)
        except cf_api.CfApiError as exc:
            # Roll back so the portal never claims access that Access won't grant.
            db.remove_assignment(device_id, email)
            db.audit(g.email, "cf_error", device_id, op="assign", error=str(exc))
            abort(500, description="Cloudflare policy update failed: %s" % exc)
        db.audit(g.email, "assign", device_id, email=email)
        return redirect(url_for("admin"))

    @app.post("/admin/devices/<device_id>/unassign")
    @require_admin
    def unassign(device_id):
        check_csrf()
        device = db.get_device(device_id) or abort(404)
        email = request.form.get("email", "").strip().lower()
        db.remove_assignment(device_id, email)
        try:
            _sync_access_policy(device)
        except cf_api.CfApiError as exc:
            db.add_assignment(device_id, email, g.email)  # restore consistency
            db.audit(g.email, "cf_error", device_id, op="unassign", error=str(exc))
            abort(500, description="Cloudflare policy update failed: %s" % exc)
        db.audit(g.email, "unassign", device_id, email=email)
        return redirect(url_for("admin"))

    @app.post("/admin/devices/<device_id>/delete")
    @require_admin
    def delete_device(device_id):
        check_csrf()
        device = db.get_device(device_id) or abort(404)
        # Best-effort teardown in reverse creation order; report what failed
        # but always remove the DB row last only if everything CF-side is gone.
        failures = []
        for op, call in (
            ("dns", lambda: cf_api.delete_dns_record(device["dns_record_id"])),
            ("tunnel", lambda: cf_api.delete_tunnel(device["tunnel_id"])),
            ("access", lambda: cf_api.delete_access_app(device["access_app_id"])),
        ):
            try:
                call()
            except cf_api.CfApiError as exc:
                failures.append("%s: %s" % (op, exc))
        if failures:
            db.audit(g.email, "cf_error", device_id, op="delete", error="; ".join(failures))
            abort(500, description="Decommission incomplete — fix in the Cloudflare "
                                   "dashboard, then retry. Failed: %s" % "; ".join(failures))
        db.delete_device(device_id)
        db.audit(g.email, "delete", device_id, hostname=device["hostname"])
        return redirect(url_for("admin"))

    # ── Provisioning API (bench script; not a browser client) ────────────────
    #
    # Two gates: a Cloudflare Access service-token policy at the edge (the
    # script sends CF-Access-Client-Id/Secret), and this bearer token, so a
    # leaked service token alone still can't register devices.

    @app.post("/api/provision")
    def provision():
        auth_header = request.headers.get("Authorization", "")
        expected = "Bearer %s" % (config.PROVISION_TOKEN or "-")
        if not hmac.compare_digest(auth_header, expected):
            abort(401)

        body = request.get_json(silent=True) or {}
        device_id = str(body.get("device_id", "")).strip().lower()
        notes = str(body.get("notes", ""))[:500]
        if not _DEVICE_ID_RE.match(device_id):
            abort(400, description="device_id must be 2-31 chars of [a-z0-9-]")

        hostname = config.device_hostname(device_id)

        # Idempotent re-run: an already-registered device just gets its tunnel
        # token re-issued (e.g. the bench script died after registration).
        existing = db.get_device(device_id)
        if existing:
            token = cf_api.get_tunnel_token(existing["tunnel_id"])
            db.audit("provisioner", "provision", device_id, rerun=True)
            return jsonify(device_id=device_id, hostname=existing["hostname"],
                           tunnel_token=token, existing=True)

        # Order matters: the Access app must exist before the DNS record so the
        # hostname is never publicly reachable ungated. New devices allow
        # admins only until an admin assigns a customer.
        created = {}  # for rollback
        try:
            app_id, policy_id = cf_api.create_access_app(
                hostname, config.ADMIN_EMAILS or ["placeholder@invalid"])
            created["access_app_id"] = app_id
            tunnel_id = cf_api.create_tunnel("sim-%s" % device_id)
            created["tunnel_id"] = tunnel_id
            cf_api.put_tunnel_config(tunnel_id, hostname)
            dns_record_id = cf_api.create_dns_cname(
                "sim-%s" % device_id, "%s.cfargotunnel.com" % tunnel_id)
            created["dns_record_id"] = dns_record_id
            token = cf_api.get_tunnel_token(tunnel_id)
        except cf_api.CfApiError as exc:
            _rollback_provision(created)
            db.audit("provisioner", "cf_error", device_id, op="provision", error=str(exc))
            abort(500, description="Cloudflare provisioning failed (rolled back): %s" % exc)

        db.insert_device(device_id, hostname, tunnel_id, app_id, policy_id,
                         dns_record_id, notes)
        db.audit("provisioner", "provision", device_id, hostname=hostname)
        return jsonify(device_id=device_id, hostname=hostname,
                       tunnel_token=token, existing=False)

    def _rollback_provision(created):
        if "dns_record_id" in created:
            try:
                cf_api.delete_dns_record(created["dns_record_id"])
            except cf_api.CfApiError:
                pass
        if "tunnel_id" in created:
            try:
                cf_api.delete_tunnel(created["tunnel_id"])
            except cf_api.CfApiError:
                pass
        if "access_app_id" in created:
            try:
                cf_api.delete_access_app(created["access_app_id"])
            except cf_api.CfApiError:
                pass

    # ── Errors ────────────────────────────────────────────────────────────────

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(500)
    def error_page(err):
        if request.path.startswith("/api/"):
            return jsonify(error=err.description or err.name), err.code
        return render_template("error.html", error=err), err.code

    return app
