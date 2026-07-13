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

import hashlib
import hmac
import os
import re
import secrets
import threading
import time

from flask import (Flask, abort, g, jsonify, redirect, render_template,
                   request, send_file, session, url_for)

from . import cf_api, config, db, status
from .auth import require_admin, require_customer

_DEVICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app():
    config.validate()
    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY or secrets.token_hex(32)
    # Device video uploads are the only large requests; everything else is
    # tiny forms, so one global cap (file + multipart overhead) is fine.
    app.config["MAX_CONTENT_LENGTH"] = config.VIDEO_MAX_UPLOAD_BYTES + 1024 * 1024
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
                "emails": [{"email": e, "by_device": by == "device"}
                           for (e, by) in db.get_assignments(d["device_id"])],
                "status": statuses.get(d["tunnel_id"], "unknown"),
                "online": status.is_online(statuses.get(d["tunnel_id"])),
            })
        return rows

    @app.get("/admin")
    @require_admin
    def admin():
        return render_template("admin.html", devices=_admin_rows(), email=g.email)

    def _cf_disabled():
        """Local development without a Cloudflare account: the DB is the only
        state, so the provision/claim/assign flows can run fully offline."""
        return config.DEV_MODE and not config.CF_API_TOKEN

    def _sync_access_policy(device):
        """Push the device's allowed-email set (customers + admins) to its
        Cloudflare Access policy. Admins stay on every policy for support."""
        emails = set(db.device_emails(device["device_id"])) | set(config.ADMIN_EMAILS)
        if _cf_disabled():
            app.logger.info("DEV_MODE: skipping Access policy sync for %s: %s",
                            device["device_id"], sorted(emails))
            return
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
        # The videos cascade away with the device row; remove their files too.
        for video in db.videos_for_device(device_id):
            _remove_video(video, g.email)
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

        # Every provision (first or re-run) mints a fresh claim secret; the Pi
        # stores it and later uses it to authenticate /api/device requests.
        # Only its sha256 is kept here, so a stolen portal DB can't claim.
        claim_secret = secrets.token_urlsafe(32)
        claim_hash = hashlib.sha256(claim_secret.encode()).hexdigest()

        # Idempotent re-run: an already-registered device just gets its tunnel
        # token re-issued (e.g. the bench script died after registration). The
        # claim secret rotates — re-provisioning means the device is in hand,
        # and whatever secret older images hold stops working.
        existing = db.get_device(device_id)
        if existing:
            token = ("dev-tunnel-token" if _cf_disabled()
                     else cf_api.get_tunnel_token(existing["tunnel_id"]))
            db.set_claim_secret_hash(device_id, claim_hash)
            db.audit("provisioner", "provision", device_id, rerun=True,
                     secret_rotated=True)
            return jsonify(device_id=device_id, hostname=existing["hostname"],
                           tunnel_token=token, claim_secret=claim_secret,
                           existing=True)

        # Order matters: the Access app must exist before the DNS record so the
        # hostname is never publicly reachable ungated. New devices allow
        # admins only until an admin assigns a customer.
        if _cf_disabled():
            db.insert_device(device_id, hostname, "dev-tun-" + device_id,
                             "dev-app", "dev-pol", "dev-dns", notes,
                             claim_secret_hash=claim_hash)
            db.audit("provisioner", "provision", device_id, hostname=hostname,
                     dev_mode=True)
            return jsonify(device_id=device_id, hostname=hostname,
                           tunnel_token="dev-tunnel-token",
                           claim_secret=claim_secret, existing=False)
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
                         dns_record_id, notes, claim_secret_hash=claim_hash)
        db.audit("provisioner", "provision", device_id, hostname=hostname)
        return jsonify(device_id=device_id, hostname=hostname,
                       tunnel_token=token, claim_secret=claim_secret,
                       existing=False)

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

    # ── Device API (shipped Pis) ──────────────────────────────────────────────
    #
    # Called by the FluoroSim Remote Access tab on customer devices. Shipped
    # Pis have neither an Access service token nor a browser session, so these
    # paths sit behind a path-scoped Access application with a Bypass policy
    # (see portal/README.md); the real gate is the per-device claim secret
    # minted at provisioning, plus a small rate limit against guessing.

    _rl_lock = threading.Lock()
    _rl_hits = {}  # key -> [timestamps]
    RL_MAX, RL_WINDOW = 10, 300.0

    def _rate_limit(*keys):
        now = time.monotonic()
        with _rl_lock:
            for key in keys:
                hits = [t for t in _rl_hits.get(key, []) if now - t < RL_WINDOW]
                if len(hits) >= RL_MAX:
                    _rl_hits[key] = hits
                    abort(429, description="Too many requests — try again later.")
                hits.append(now)
                _rl_hits[key] = hits

    def _authenticate_device(body):
        """Return the device row iff body carries its valid claim secret.

        Uniform 403 whether the device id is unknown, has no secret on file
        (pre-rotation record), or the secret is wrong — no existence leak.
        """
        device_id = str(body.get("device_id", "")).strip().lower()
        secret = str(body.get("claim_secret", ""))
        _rate_limit("ip:%s" % request.remote_addr, "dev:%s" % device_id)
        device = db.get_device(device_id) if _DEVICE_ID_RE.match(device_id) else None
        stored = device["claim_secret_hash"] if device else None
        presented = hashlib.sha256(secret.encode()).hexdigest()
        if not stored or not hmac.compare_digest(presented, stored):
            abort(403, description="Unknown device or bad claim secret.")
        return device

    def _device_claimed_email(device_id):
        emails = [e for (e, by) in db.get_assignments(device_id) if by == "device"]
        return emails[0] if emails else None

    @app.post("/api/device/status")
    def device_status():
        # POST, not GET: the claim secret must never appear in a URL or log.
        body = request.get_json(silent=True) or {}
        device = _authenticate_device(body)
        return jsonify(device_id=device["device_id"], hostname=device["hostname"],
                       claimed_email=_device_claimed_email(device["device_id"]))

    @app.post("/api/device/claim")
    def device_claim():
        body = request.get_json(silent=True) or {}
        device = _authenticate_device(body)
        device_id = device["device_id"]
        email = str(body.get("email", "")).strip().lower()
        if not _EMAIL_RE.match(email):
            abort(400, description="That doesn't look like an email address.")
        removed = db.replace_device_claim(device_id, email)
        try:
            _sync_access_policy(device)
        except cf_api.CfApiError as exc:
            # Same invariant as admin assign: never claim access that
            # Cloudflare Access won't actually grant.
            db.remove_assignment(device_id, email)
            db.restore_assignments(device_id, removed)
            db.audit("device:%s" % device_id, "cf_error", device_id,
                     op="claim", error=str(exc))
            abort(500, description="Could not update remote access — try again.")
        db.audit("device:%s" % device_id, "claim", device_id, email=email,
                 replaced=[e for (e, _by) in removed])
        return jsonify(device_id=device_id, hostname=device["hostname"],
                       claimed_email=email)

    @app.post("/api/device/videos")
    def device_upload_video():
        """Auto-upload of a finished session recording (multipart form).

        Same gate as the other /api/device endpoints — the per-device claim
        secret — but read from form fields, since the video rides along as a
        file part. The client sends the file's sha256; uploads are deduped on
        (device, sha256) so a retry after a dropped response is a no-op.
        """
        device = _authenticate_device(request.form)
        device_id = device["device_id"]

        sha_claim = str(request.form.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha_claim):
            abort(400, description="Missing or malformed sha256.")
        upload = request.files.get("video")
        if upload is None:
            abort(400, description="Missing 'video' file part.")

        existing = db.find_video_by_sha(device_id, sha_claim)
        if existing:
            return jsonify(video_id=existing["video_id"], duplicate=True)

        # content_length includes multipart overhead — a slightly conservative
        # quota check is fine (we'd rather refuse at 99.9% than overrun).
        if db.video_bytes_total() + (request.content_length or 0) > config.VIDEO_QUOTA_BYTES:
            db.audit("device:%s" % device_id, "video_quota_full", device_id)
            # Plain return, not abort(): werkzeug has no default 507 exception.
            return jsonify(error="Server video storage is full — delete old "
                                 "videos in the portal library."), 507

        try:
            duration_s = max(0, int(float(request.form.get("duration_s", 0))))
        except ValueError:
            duration_s = 0
        try:
            recorded_epoch = float(request.form.get("recorded_at", 0)) or time.time()
        except ValueError:
            recorded_epoch = time.time()
        recorded_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(recorded_epoch))

        # Stream to disk hashing as we go; keep the file only if the digest
        # matches what the device claimed (a truncated body must not dedupe
        # differently on retry, or the good copy could never be uploaded).
        video_id = secrets.token_urlsafe(16)
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        path = _video_path(video_id)
        digest = hashlib.sha256()
        size = 0
        with open(path, "wb") as out:
            while True:
                chunk = upload.stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
        if size == 0 or digest.hexdigest() != sha_claim:
            os.remove(path)
            abort(400, description="Upload corrupted in transit (checksum "
                                   "mismatch) — retry.")

        title = "Session %s" % time.strftime("%Y-%m-%d %H:%M",
                                             time.gmtime(recorded_epoch))
        try:
            db.insert_video(video_id, device_id, title, sha_claim, size,
                            duration_s, recorded_at)
        except Exception:
            # Lost a same-sha race with a concurrent retry; keep that row.
            os.remove(path)
            existing = db.find_video_by_sha(device_id, sha_claim)
            if not existing:
                raise
            return jsonify(video_id=existing["video_id"], duplicate=True)
        db.audit("device:%s" % device_id, "video_upload", device_id,
                 video_id=video_id, size=size)
        return jsonify(video_id=video_id, duplicate=False)

    # ── Video library (customer pages) ────────────────────────────────────────
    #
    # A video belongs to its device: everyone currently assigned to the device
    # may manage it (rename/share/delete); video_shares grants view-only access
    # to any signed-in email. Admins may do everything.

    def _video_path(video_id):
        return os.path.join(config.VIDEO_DIR, video_id + ".mp4")

    def _get_video_or_404(video_id):
        # The id is a token we minted, but never let a crafted one traverse
        # out of VIDEO_DIR anyway.
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", video_id):
            abort(404)
        return db.get_video(video_id) or abort(404)

    def _can_manage(video):
        return g.is_admin or g.email in db.device_emails(video["device_id"])

    def _can_view(video):
        return _can_manage(video) or g.email in db.video_shares(video["video_id"])

    def _remove_video(video, actor):
        try:
            os.remove(_video_path(video["video_id"]))
        except OSError:
            pass  # row is authoritative; a missing file just means less disk
        db.delete_video(video["video_id"])
        db.audit(actor, "video_delete", video["device_id"],
                 video_id=video["video_id"], title=video["title"])

    @app.get("/library")
    @require_customer
    def library_page():
        rows = db.all_videos() if g.is_admin else db.videos_for_email(g.email)
        videos = []
        for v in rows:
            manage = g.is_admin or (v["via"] if "via" in v.keys() else "device") == "device"
            videos.append({
                "video_id": v["video_id"],
                "device_id": v["device_id"],
                "title": v["title"],
                "size": v["size"],
                "duration_s": v["duration_s"],
                "recorded_at": v["recorded_at"],
                "manage": manage,
                "shares": db.video_shares(v["video_id"]) if manage else [],
            })
        return render_template("library.html", videos=videos, email=g.email,
                               is_admin=g.is_admin)

    @app.get("/videos/<video_id>/stream")
    @require_customer
    def video_stream(video_id):
        video = _get_video_or_404(video_id)
        if not _can_view(video):
            abort(403)
        path = _video_path(video_id)
        if not os.path.exists(path):
            abort(404)
        # conditional=True gives Range support, which <video> needs to seek.
        return send_file(path, mimetype="video/mp4", conditional=True,
                         download_name="%s.mp4" % video["title"])

    @app.post("/videos/<video_id>/title")
    @require_customer
    def video_title(video_id):
        check_csrf()
        video = _get_video_or_404(video_id)
        if not _can_manage(video):
            abort(403)
        title = request.form.get("title", "").strip()
        if not 1 <= len(title) <= 120:
            abort(400, description="Titles are 1-120 characters.")
        db.set_video_title(video_id, title)
        db.audit(g.email, "video_title", video["device_id"],
                 video_id=video_id, title=title)
        return redirect(url_for("library_page"))

    @app.post("/videos/<video_id>/share")
    @require_customer
    def video_share(video_id):
        check_csrf()
        video = _get_video_or_404(video_id)
        if not _can_manage(video):
            abort(403)
        share_email = request.form.get("email", "").strip().lower()
        if not _EMAIL_RE.match(share_email):
            abort(400, description="That doesn't look like an email address.")
        db.add_video_share(video_id, share_email, g.email)
        db.audit(g.email, "video_share", video["device_id"],
                 video_id=video_id, email=share_email)
        return redirect(url_for("library_page"))

    @app.post("/videos/<video_id>/unshare")
    @require_customer
    def video_unshare(video_id):
        check_csrf()
        video = _get_video_or_404(video_id)
        share_email = request.form.get("email", "").strip().lower()
        # Owners revoke anyone; a share recipient may remove themselves.
        if not (_can_manage(video) or share_email == g.email):
            abort(403)
        db.remove_video_share(video_id, share_email)
        db.audit(g.email, "video_unshare", video["device_id"],
                 video_id=video_id, email=share_email)
        return redirect(url_for("library_page"))

    @app.post("/videos/<video_id>/delete")
    @require_customer
    def video_delete(video_id):
        check_csrf()
        video = _get_video_or_404(video_id)
        if not _can_manage(video):
            abort(403)
        _remove_video(video, g.email)
        return redirect(url_for("library_page"))

    # ── Errors ────────────────────────────────────────────────────────────────

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    @app.errorhandler(429)
    @app.errorhandler(500)
    def error_page(err):
        if request.path.startswith("/api/"):
            return jsonify(error=err.description or err.name), err.code
        return render_template("error.html", error=err), err.code

    return app
