# Pi setup — manufacturing checklist

Turns a freshly imaged Raspberry Pi into a customer-ready FluoroSim device:
FluoroSim supervised by systemd, plus a Cloudflare Tunnel that makes the web
panel reachable at `https://sim-<device-id>.<device-domain>/`, gated by
Cloudflare Access (only the assigned customer + ReSuture admins can open it).

## Prerequisites

- Raspberry Pi OS 64-bit with desktop autologin (standard image), camera
  attached. Runs on a Pi 3 B+ or a Pi 5.
- This repo cloned at `~/fluoro_simulator`
  (`git clone https://github.com/ReSuture/fluoro_simulator.git`).
- The provision script installs the app's packages from apt:
  `python3-opencv python3-numpy python3-flask ffmpeg wl-mirror wlr-randr`. Do not
  `pip install` them: Raspberry Pi OS blocks system-wide pip installs.
- From the portal operator (see `portal/README.md`): the portal URL, the
  `PROVISION_TOKEN`, and the Access service-token id/secret for the bench.

### Pi 5 hardware

- **USB power.** Use the official 27 W (5 A) supply. On a weaker supply the
  Pi 5 caps all four USB ports at 600 mA combined, and two cameras plus the
  touch panel and pedal can exceed that, so cameras drop out. To lift the cap
  on a supply you trust, set `usb_max_current_enable=1` in
  `/boot/firmware/config.txt`.
- **Touch panel.** Either HDMI port works, because the panel is found by its EDID
  name. Its touch cable goes to a Pi USB port; that is what makes touch work.
- **Second monitor (optional).** Plug it into the other HDMI port and it mirrors
  the panel (see the main README, *Touch panel and a mirror monitor*).
- Fit the active cooler.

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
2. Installs the app's apt packages, and `cloudflared` if missing.
3. Registers the device with the portal, which creates the tunnel, DNS record
   and Access application **in that order's safe variant** (Access gate first),
   and returns a token scoped to this one tunnel.
4. Installs the `cloudflared` system service with that token.
5. Installs `fluorosim.service` (auto-start + crash restart) and
   `fluorosim-mirror.service` (second-monitor mirror) as user services, and
   enables lingering so they run from boot.
6. Verifies `https://sim-<id>.<device-domain>/` answers with a redirect to the
   Access login. **If it answers 200 without a login, do not ship** — the gate
   is missing.

The script is idempotent — re-run it after any failure.

It leaves three files in `/etc/fluorosim/`:

| File | Mode | Contents |
|---|---|---|
| `device-id` | 644 root | the 6-char device id |
| `portal-url` | 644 root | `https://portal.<device-domain>` |
| `claim-secret` | **600, desktop user** | per-device secret for `/api/device/*` |

The claim secret lets the customer self-register their email from the app's
**Remote Access** tab (WiFi → email → done). **Every re-provision rotates
it** — old images of the device can no longer claim; that's intended.

The same secret authenticates the **Library auto-upload**: when a Record
session ends, the app transcodes it to MP4 and uploads it to the portal's
`/library`. The Library tab shows a dot per recording — green (in the
portal library), amber (upload pending/retrying), red (failed), grey (local
only, e.g. unprovisioned).

Afterwards the device shows up as **unassigned** on the portal's `/admin`
page; the customer self-registers from the Remote Access tab, or assign
their email at `/admin` when the unit ships.

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
