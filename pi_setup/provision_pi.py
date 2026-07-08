#!/usr/bin/env python3
"""One-shot bench provisioning for a FluoroSim Raspberry Pi.

Registers this Pi with the ReSuture portal, which creates its Cloudflare
Tunnel + DNS + Access gate and returns a single-tunnel token; then installs
cloudflared with that token and enables the FluoroSim user service.

Run as the desktop user (it uses sudo where needed):

    python3 provision_pi.py --portal https://portal.<device-domain> \
        --provision-token <PROVISION_TOKEN> \
        --cf-client-id <SERVICE_TOKEN_ID> --cf-client-secret <SERVICE_TOKEN_SECRET> \
        [--device-id abc123]

The three secrets come from the portal operator (see portal/README.md):
PROVISION_TOKEN from /etc/resuture-portal.env, and the Access service-token
pair from Zero Trust -> Access -> Service Auth. None of them persist on the
Pi; the only credential left behind is the tunnel token inside the cloudflared
service, which is scoped to this one device's tunnel.

Idempotent: re-running re-issues the same device's tunnel token and refreshes
the local services.
"""

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request

CLOUDFLARED_DEB = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-linux-{arch}.deb")
DEVICE_ID_FILE = "/etc/fluorosim/device-id"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print("\033[92m[provision]\033[0m %s" % msg)


def fail(msg):
    print("\033[91m[provision] ERROR:\033[0m %s" % msg, file=sys.stderr)
    sys.exit(1)


def run(cmd, **kwargs):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


# ── Device id ─────────────────────────────────────────────────────────────────

def default_device_id():
    """Last 6 hex chars of the Pi's CPU serial — stable across re-imaging and
    short enough for a sticker. Falls back to random if unreadable."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                m = re.match(r"Serial\s*:\s*([0-9a-fA-F]+)", line)
                if m:
                    return m.group(1)[-6:].lower()
    except OSError:
        pass
    return secrets.token_hex(3)


# ── cloudflared ───────────────────────────────────────────────────────────────

def install_cloudflared():
    if shutil.which("cloudflared"):
        log("cloudflared already installed")
        return
    machine = platform.machine()
    arch = "arm64" if machine == "aarch64" else "armhf" if machine.startswith("arm") else "amd64"
    deb = "/tmp/cloudflared.deb"
    url = CLOUDFLARED_DEB.format(arch=arch)
    log("downloading %s" % url)
    urllib.request.urlretrieve(url, deb)
    run(["sudo", "dpkg", "-i", deb])
    os.remove(deb)


# ── Portal registration ───────────────────────────────────────────────────────

def register_with_portal(args, device_id):
    req = urllib.request.Request(
        args.portal.rstrip("/") + "/api/provision",
        data=json.dumps({"device_id": device_id, "notes": args.notes}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % args.provision_token,
            "CF-Access-Client-Id": args.cf_client_id,
            "CF-Access-Client-Secret": args.cf_client_secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        fail("portal rejected provisioning (HTTP %d): %s" % (e.code, e.read().decode()[:500]))
    except urllib.error.URLError as e:
        fail("cannot reach the portal at %s: %s" % (args.portal, e.reason))


# ── Services ──────────────────────────────────────────────────────────────────

def install_tunnel_service(tunnel_token):
    # `cloudflared service install` writes /etc/systemd/system/cloudflared.service
    # itself. If a previous install exists, refresh it.
    if os.path.exists("/etc/systemd/system/cloudflared.service"):
        run(["sudo", "cloudflared", "service", "uninstall"])
    run(["sudo", "cloudflared", "service", "install", tunnel_token])
    run(["sudo", "systemctl", "enable", "--now", "cloudflared"])


def install_fluorosim_service():
    unit_src = os.path.join(REPO_DIR, "pi_setup", "fluorosim.service")
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    shutil.copy(unit_src, os.path.join(unit_dir, "fluorosim.service"))
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", "--now", "fluorosim.service"])
    # Let the user services run without an interactive login.
    run(["sudo", "loginctl", "enable-linger", os.environ.get("USER", "")])


def write_device_id(device_id):
    run(["sudo", "mkdir", "-p", os.path.dirname(DEVICE_ID_FILE)])
    subprocess.run(
        ["sudo", "tee", DEVICE_ID_FILE],
        input=device_id + "\n", text=True, check=True,
        stdout=subprocess.DEVNULL,
    )


# ── Verification ──────────────────────────────────────────────────────────────

def verify(hostname):
    """The hostname must answer with a redirect to the Access login — that
    proves both the tunnel is up AND the gate is in front of it."""
    url = "https://%s/" % hostname
    log("verifying %s (waiting for the tunnel to connect)" % url)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            opener = urllib.request.build_opener(_NoRedirect())
            resp = opener.open(req, timeout=10)
            code = resp.getcode()
        except urllib.error.HTTPError as e:
            code = e.code
        except urllib.error.URLError:
            time.sleep(5)
            continue
        if code in (301, 302, 303, 307):
            log("OK — %s redirects to the Cloudflare Access login" % hostname)
            return True
        if code == 200:
            fail("%s answered 200 WITHOUT an Access login — the hostname is "
                 "publicly reachable ungated. Do not ship this device; check "
                 "the Access application in the Cloudflare dashboard." % hostname)
        time.sleep(5)
    log("WARNING: %s did not answer within 2 minutes. The tunnel may still be "
        "connecting; check `systemctl status cloudflared` and retry." % url)
    return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--portal", required=True, help="https://portal.<device-domain>")
    p.add_argument("--provision-token", required=True)
    p.add_argument("--cf-client-id", required=True,
                   help="Access service token id (CF-Access-Client-Id)")
    p.add_argument("--cf-client-secret", required=True,
                   help="Access service token secret")
    p.add_argument("--device-id", default=None,
                   help="override the id derived from the CPU serial")
    p.add_argument("--notes", default="", help="free-text note shown in /admin")
    args = p.parse_args()

    if os.geteuid() == 0:
        fail("run as the desktop user, not root (sudo is used where needed)")

    device_id = (args.device_id or default_device_id()).lower()
    log("device id: %s" % device_id)

    install_cloudflared()

    result = register_with_portal(args, device_id)
    hostname = result["hostname"]
    log("registered with portal: %s%s"
        % (hostname, " (existing device, token re-issued)" if result.get("existing") else ""))

    install_tunnel_service(result["tunnel_token"])
    install_fluorosim_service()
    write_device_id(device_id)
    verify(hostname)

    print()
    log("─" * 60)
    log("Device ID : %s   (put this on the sticker)" % device_id)
    log("Panel URL : https://%s/" % hostname)
    log("Status    : unassigned — assign a customer at %s/admin" % args.portal.rstrip("/"))
    log("─" * 60)


if __name__ == "__main__":
    main()
