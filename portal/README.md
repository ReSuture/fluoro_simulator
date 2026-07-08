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
| `ACCESS_TEAM_DOMAIN` | `<team>.cloudflareaccess.com` (Zero Trust → Settings → Custom Pages) |
| `ACCESS_PORTAL_AUD` | AUD tag of the portal's own Access application |
| `ADMIN_EMAILS` | `ben@resuture.com` (comma-separated; admins see /admin and stay on every device policy) |
| `PROVISION_TOKEN` | `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `DATABASE_PATH` | `/var/lib/resuture-portal/portal.db` |
| `SIM_DOMAIN_SUFFIX` | `.resuturesim.com` (leading dot) |
| `FLASK_SECRET_KEY` | another `secrets.token_urlsafe(32)` |
| `PORTAL_DEV` | `1` only for local development — enables the `X-Dev-Email` auth bypass. **Never set in production.** |

## One-time Cloudflare setup (Phase 0)

1. **Buy the device domain** (e.g. `resuturesim.com`) and add it as a zone on a
   free Cloudflare plan; point the domain's nameservers at Cloudflare (this is
   a brand-new domain, so nothing can break).
2. **Zero Trust team**: dashboard → Zero Trust → choose a team name. Note the
   team domain `<team>.cloudflareaccess.com`.
3. **Login methods**: Zero Trust → Settings → Authentication → enable
   **One-time PIN** (works for every customer email; optionally add Google).
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
3. Zero Trust → Access → Applications → Add self-hosted app
   `portal.<device-domain>`, session 24h, policy **Allow** → Include →
   *Everyone with a valid login method* (or restrict to known customer emails).
   Copy the app's **AUD tag** into `ACCESS_PORTAL_AUD` and restart the service.
4. **Provisioning service token**: Zero Trust → Access → Service Auth →
   Create service token `pi-bench`. Then add a second Access application
   scoped to path `portal.<device-domain>/api/provision` with a **Service
   Auth** policy accepting that token. The bench script presents the token's
   `CF-Access-Client-Id/Secret` headers *plus* the `PROVISION_TOKEN` bearer.

## Day-to-day flow

1. **Bench**: run `pi_setup/provision_pi.py` on a new Pi → it registers itself
   and appears on `/admin` as *unassigned* (only admins can open it).
2. **Assign**: on `/admin`, enter the customer's email next to the device.
   This edits the device's Cloudflare Access policy; the customer can sign in
   immediately at the portal and open their simulator.
3. **Unassign / Decommission**: same page. Note: unassigning removes the
   policy entry, but an already-signed-in session lasts until its 24h expiry.
   To cut access instantly, also use Zero Trust → My Team → Users → Revoke
   sessions for that user.

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
