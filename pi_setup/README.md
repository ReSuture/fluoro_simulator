# Pi setup — manufacturing checklist

Turns a freshly imaged Raspberry Pi into a customer-ready FluoroSim device:
FluoroSim supervised by systemd, plus a Cloudflare Tunnel that makes the web
panel reachable at `https://sim-<device-id>.<device-domain>/`, gated by
Cloudflare Access (only the assigned customer + ReSuture admins can open it).

## Prerequisites

- Raspberry Pi OS with desktop autologin (standard image), camera attached.
- This repo cloned at `~/fluoro_simulator`.
- App dependencies: `pip install opencv-python numpy flask`.
- From the portal operator (see `portal/README.md`): the portal URL, the
  `PROVISION_TOKEN`, and the Access service-token id/secret for the bench.

## Provision

```bash
cd ~/fluoro_simulator/pi_setup
python3 provision_pi.py \
    --portal https://portal.<device-domain> \
    --provision-token <PROVISION_TOKEN> \
    --cf-client-id <SERVICE_TOKEN_ID> \
    --cf-client-secret <SERVICE_TOKEN_SECRET>
```

What it does:

1. Derives the **device id** from the Pi's CPU serial (last 6 hex chars —
   stable across re-imaging). Print it on the case sticker.
2. Installs `cloudflared` if missing.
3. Registers the device with the portal, which creates the tunnel, DNS record
   and Access application **in that order's safe variant** (Access gate first),
   and returns a token scoped to this one tunnel.
4. Installs the `cloudflared` system service with that token.
5. Installs `fluorosim.service` as a user service (auto-start + crash restart)
   and enables lingering so it runs from boot.
6. Verifies `https://sim-<id>.<device-domain>/` answers with a redirect to the
   Access login. **If it answers 200 without a login, do not ship** — the gate
   is missing.

The script is idempotent — re-run it after any failure.

Afterwards the device shows up as **unassigned** on the portal's `/admin`
page; assign the customer's email there when the unit ships.

## Security rules for tunneled Pis (do not break these)

The FluoroSim panel has **no authentication of its own**. Its safety relies on
exactly two things:

1. `launch_fluoro.sh` starts the panel with `--host 127.0.0.1` — only the
   local `cloudflared` can reach it. **Never** change this to `0.0.0.0` on a
   customer device, and never port-forward :5000.
2. Cloudflare Access gates the public hostname. **Never** un-proxy
   (grey-cloud) the DNS record, add a tunnel ingress that bypasses the
   hostname, or delete the Access application while the tunnel is live.

Also: the tunnel token embedded in the cloudflared service can only
impersonate this one device's tunnel. Decommissioning the device from
`/admin` revokes it.

## Bench testing on a LAN (non-customer device)

For a demo Pi on your own network without the tunnel, run the panel the old
way — `python3 fluoro_web.py` binds 0.0.0.0:5000 by default and stays
LAN-only. The `--host 127.0.0.1 --http` flags in `launch_fluoro.sh` are what
make a device tunnel-ready.
