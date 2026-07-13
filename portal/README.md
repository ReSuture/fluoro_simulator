# ReSuture FluoroSim Portal

A small Flask app: customers sign in (via Cloudflare Access — one-time email
PIN and/or Google, no passwords to manage) and get an "Open my simulator"
button that takes them to the FluoroSim panel on **their own Raspberry Pi**,
reachable from anywhere through a Cloudflare Tunnel.

```
Customer browser
   │  (Cloudflare Access sign-in)
   ├─► portal.<device-domain>  ── tunnel ──► this app (gunicorn :8080) on the AWS box
   │        ├─ SQLite: devices, assignments, audit log
   │        └─ Cloudflare API token (lives ONLY here)
   └─► sim-<id>.<device-domain> ── tunnel ──► http://127.0.0.1:5000 on that customer's Pi
            └─ per-device Access app; policy = that customer's email(s) + admins
```

- The device domain is a **separate domain** (e.g. `resuturesim.com`) hosted on
  Cloudflare. `resuture.com` (WordPress) is untouched — it just links to the portal.
- Every Cloudflare API operation happens in this app. The Pi provisioning
  script only ever talks to `POST /api/provision`.
- The Pi panel itself has **no authentication**: its security is (a) it binds
  127.0.0.1 so only the local `cloudflared` can reach it, and (b) Cloudflare
  Access gates the tunnel hostname. Never weaken either.

## Environment variables (`/etc/resuture-portal.env`, chmod 600)

| Variable | Example / meaning |
|---|---|
| `CF_API_TOKEN` | API token, scopes: Zone/DNS/Edit (device zone only), Account/Access: Apps and Policies/Edit, Account/Cloudflare Tunnel/Edit |
| `CF_ACCOUNT_ID` | Cloudflare account id (dashboard → Overview, right column) |
| `CF_ZONE_ID` | Zone id of the **device domain** |
| `ACCESS_TEAM_DOMAIN` | `<team>.cloudflareaccess.com` (Zero Trust → Reusable components → Custom pages → Team domain) |
| `ACCESS_PORTAL_AUD` | AUD tag of the portal's own Access application |
| `ADMIN_EMAILS` | `ben@resuture.com` (comma-separated; admins see /admin and stay on every device policy) |
| `PROVISION_TOKEN` | `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `DATABASE_PATH` | `/var/lib/resuture-portal/portal.db` |
| `SIM_DOMAIN_SUFFIX` | `.resuturesim.com` (leading dot) |
| `FLASK_SECRET_KEY` | another `secrets.token_urlsafe(32)` |
| `VIDEO_DIR` | *(optional)* uploaded session videos; default `videos/` next to the DB |
| `VIDEO_MAX_UPLOAD_MB` | *(optional, default 95)* per-file cap — keep under Cloudflare's 100 MB free-plan request limit |
| `VIDEO_QUOTA_GB` | *(optional, default 20)* total video storage; uploads are refused (HTTP 507) beyond it |
| `PORTAL_DEV` | `1` only for local development — enables the `X-Dev-Email` auth bypass. **Never set in production.** |

## One-time Cloudflare setup (Phase 0)

1. **Buy the device domain** (e.g. `resuturesim.com`) and add it as a zone on a
   free Cloudflare plan; point the domain's nameservers at Cloudflare (this is
   a brand-new domain, so nothing can break).
2. **Zero Trust team**: dashboard → Zero Trust → choose a team name. Note the
   team domain `<team>.cloudflareaccess.com`.
3. **Login methods**: Zero Trust → Integrations → Identity providers →
   Add new identity provider → **One-time PIN** (works for every customer
   email — no Cloudflare or Google account needed; optionally add Google from
   the same screen).
4. **API token**: My Profile → API Tokens → Create Token with the three scopes
   in the table above, zone-scoped to the device domain.

## Deploying the portal on the AWS box

```bash
sudo git clone <repo> /opt/fluoro_simulator
cd /opt/fluoro_simulator/portal
sudo python3 -m venv venv && sudo venv/bin/pip install -r requirements.txt
sudo useradd -r -s /usr/sbin/nologin portal
sudo mkdir -p /var/lib/resuture-portal && sudo chown portal:portal /var/lib/resuture-portal
sudoedit /etc/resuture-portal.env        # fill in the table above; chmod 600
sudo cp portal.service /etc/systemd/system/resuture-portal.service
sudo systemctl daemon-reload && sudo systemctl enable --now resuture-portal
curl -s localhost:8080/healthz           # → {"ok": true}
```

Expose it (one-off, in the Zero Trust dashboard — this is the only tunnel not
created by the portal itself):

1. Zero Trust → Networks → Tunnels → Create tunnel, name `resuture-portal`,
   install `cloudflared` on the AWS box with the shown token.
2. Add a public hostname: `portal.<device-domain>` → `http://localhost:8080`.
   (No security-group changes — cloudflared only makes outbound connections.)
3. Zero Trust → Access controls → Applications → Add self-hosted app
   `portal.<device-domain>`, session 24h, policy **Allow** → Include →
   *Everyone with a valid login method* (or restrict to known customer emails).
   Copy the app's **AUD tag** into `ACCESS_PORTAL_AUD` and restart the service.
4. **Provisioning service token**: Zero Trust → Access controls → Service
   credentials → Service Tokens → Create Service Token, name `pi-bench`.
   Then add a second Access application scoped to path
   `portal.<device-domain>/api/provision` with a **Service Auth** policy
   accepting that token. The bench script presents the token's
   `CF-Access-Client-Id/Secret` headers *plus* the `PROVISION_TOKEN` bearer.
5. **Device API bypass**: add a third Access application scoped to path
   `portal.<device-domain>/api/device` with a single **Bypass** policy
   (Everyone). Shipped Pis call `/api/device/status` and `/api/device/claim`
   from their Remote Access tab, and they hold neither a service token nor a
   browser session — the gate for these endpoints is application-level: a
   per-device `claim_secret` minted by `/api/provision` (only its sha256 is
   stored), compared with `hmac.compare_digest`, plus a small rate limit.
   Device requests must send a real `User-Agent` header (Cloudflare's Browser
   Integrity Check rejects Python-urllib's default with error 1010).

## Day-to-day flow

1. **Bench**: run `pi_setup/provision_pi.py` on a new Pi → it registers itself
   and appears on `/admin` as *unassigned* (only admins can open it). The
   response includes a fresh `claim_secret` the Pi stores; **re-provisioning
   rotates it** (older images of that device can no longer claim — intended).
2. **Assign**: either the customer self-registers from the device's Remote
   Access tab (WiFi → enter email → `/api/device/claim`), or an admin enters
   the email on `/admin`. Both edit the device's Cloudflare Access policy; the
   customer can sign in immediately at the portal and open their simulator.
   A device claim replaces the previous device claim (one self-registered
   email per device, shown as "(device)" on `/admin`) but never touches
   admin-added assignments; it stays until re-claimed or unassigned.
3. **Unassign / Decommission**: same page. Note: unassigning removes the
   policy entry, but an already-signed-in session lasts until its 24h expiry.
   To cut access instantly, also use Zero Trust → My Team → Users → Revoke
   sessions for that user.

## Session video library

Devices auto-upload finished recordings (transcoded to H.264 MP4 on the Pi)
to `POST /api/device/videos` — the same claim-secret gate and path-scoped
Bypass Access app as the other `/api/device` endpoints, plus sha256 dedupe so
retries never duplicate. Signed-in users see them at `/library`:

- A video belongs to its **device**: every email assigned to the device can
  play, rename, **share**, and delete it. Admins can manage everything.
- Sharing grants **view-only** access to any email — the colleague signs in
  with the usual one-time-PIN flow and finds the video in their `/library`
  (no simulator required). Recipients can remove themselves.
- Storage: files live in `VIDEO_DIR`, capped by `VIDEO_QUOTA_GB`; when full,
  devices show "server storage full" until someone deletes old videos.
  Decommissioning a device deletes its videos and files.
- **Caveat**: because visibility follows the device's *current* assignments,
  reassigning a device to a new customer hands them its whole library —
  delete the videos first if that matters.

## Local development (no Cloudflare needed)

```powershell
pip install -r portal/requirements.txt
$env:PORTAL_DEV = "1"; $env:ADMIN_EMAILS = "ben@resuture.com"
python -m flask --app portal.app:create_app run --port 8080
# customer view:
curl -H "X-Dev-Email: customer@example.com" http://localhost:8080/
# admin view:
curl -H "X-Dev-Email: ben@resuture.com" http://localhost:8080/admin
```

Tests: `python -m pytest portal/tests` (JWT verification against a local
RS256 keypair; provision/assign flows with the Cloudflare API mocked).

## Deploying schema changes

`db.init_db()` migrates the SQLite database in place at boot (e.g. adding
`devices.claim_secret_hash`). Still: back up `/var/lib/resuture-portal/portal.db`
before deploying a version that changes the schema.

## Security notes

- Admin pages are double-gated: Cloudflare Access must authenticate the email
  **and** it must be in `ADMIN_EMAILS`.
- JWTs are verified against the team JWKS with the portal's own AUD — a token
  for another app in the same team is rejected. The plain
  `Cf-Access-Authenticated-User-Email` header is never trusted.
- A stolen Pi's tunnel token can only impersonate that one device's origin;
  decommissioning kills it.
- The Access application for a device hostname is always created **before**
  its DNS record, so a device URL is never live without a gate.
