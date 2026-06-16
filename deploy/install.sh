#!/usr/bin/env bash
# Crittercam appliance installer — the ONE place that touches system state.
# Run with:   sudo deploy/install.sh
#
# Idempotent: safe to re-run. Installs systemd units, pins mDNS to the LAN so
# <hostname>.local is reachable, installs the kiosk browser, and enables
# everything. Leaves the hostname alone (so whatever name you've chosen, e.g.
# rig.local, is preserved). Does NOT edit the user config (web.port / schedule) —
# that's a deliberate, separate step so it can be sequenced against the running
# processes.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root: sudo $0" >&2
    exit 1
fi

# Resolve the real (non-root) user so user-owned files land in the right home.
TARGET_USER="${SUDO_USER:-sajeel}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
echo "==> repo=$REPO  user=$TARGET_USER  home=$TARGET_HOME"

echo "==> 1/6 installing kiosk browser + helpers (flatpak Chromium, unclutter)"
# NB: the distro 'chromium-browser' is a snap, and snap-confine is broken on this
# Tegra kernel (no AppArmor) — it never launches ("cap_dac_override not found").
# We install Chromium from Flathub instead (per-user, runs under bubblewrap).
# unclutter (apt) hides the idle cursor; flatpak provides the runtime sandbox.
APT_OK=1
apt-get update -qq || APT_OK=0
for pkg in flatpak unclutter; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
        echo "    $pkg already installed"
    elif [ "$APT_OK" -eq 1 ] && apt-get install -y "$pkg"; then
        echo "    installed $pkg"
    else
        echo "    WARNING: could not install $pkg"
    fi
done
# Flathub remote + Chromium, both per-user for $TARGET_USER (no root-owned flatpak state).
if command -v flatpak >/dev/null 2>&1; then
    sudo -u "$TARGET_USER" flatpak --user remote-add --if-not-exists \
        flathub https://flathub.org/repo/flathub.flatpakrepo || true
    if sudo -u "$TARGET_USER" flatpak info org.chromium.Chromium >/dev/null 2>&1; then
        echo "    org.chromium.Chromium already installed"
    elif sudo -u "$TARGET_USER" flatpak --user install -y --noninteractive \
            flathub org.chromium.Chromium; then
        echo "    installed org.chromium.Chromium"
    else
        echo "    WARNING: could not install Chromium flatpak (kiosk will fall back to Epiphany)"
    fi
else
    echo "    WARNING: flatpak unavailable (kiosk will fall back to Epiphany)"
fi

echo "==> 2/6 installing systemd units"
install -m 644 "$REPO"/deploy/systemd/*.service "$REPO"/deploy/systemd/*.timer /etc/systemd/system/
# remove obsolete suspend resume-hook from earlier installs (camera-off keeps the box on)
rm -f /usr/lib/systemd/system-sleep/crittercam-resume

echo "==> 3/6 pinning mDNS to the LAN so $(hostname).local resolves to the WiFi IPv4"
sh "$REPO"/deploy/scripts/fix-avahi-lan.sh || echo "    WARNING: avahi pin failed (check $(hostname).local manually)"

echo "==> 4/6 verifying gdm autologin for $TARGET_USER"
GDM_CONF=/etc/gdm3/custom.conf
if grep -qE "^\s*AutomaticLoginEnable\s*=\s*[Tt]rue" "$GDM_CONF" 2>/dev/null \
   && grep -qE "^\s*AutomaticLogin\s*=\s*$TARGET_USER" "$GDM_CONF" 2>/dev/null; then
    echo "    autologin already enabled — ok"
else
    echo "    enabling autologin in $GDM_CONF"
    # Ensure a [daemon] section with the two keys (best-effort, minimal edit).
    if ! grep -q "^\[daemon\]" "$GDM_CONF" 2>/dev/null; then
        printf '\n[daemon]\n' >> "$GDM_CONF"
    fi
    sed -i -E "/^\[daemon\]/a AutomaticLoginEnable=true\nAutomaticLogin=$TARGET_USER" "$GDM_CONF"
fi

echo "==> 5/6 installing kiosk autostart for $TARGET_USER"
install -d -o "$TARGET_USER" -g "$TARGET_USER" "$TARGET_HOME/.config/autostart"
install -m 644 -o "$TARGET_USER" -g "$TARGET_USER" \
    "$REPO"/deploy/autostart/crittercam-kiosk.desktop \
    "$TARGET_HOME/.config/autostart/crittercam-kiosk.desktop"

echo "==> 6/6 enabling services"
systemctl daemon-reload
systemctl enable --now \
    crittercam-tracker.service \
    crittercam-web.service \
    crittercam-schedule.timer \
    crittercam-healthcheck.timer

cat <<EOF

==> done.

Next (one-time) — set the runtime overrides in $TARGET_HOME/wildlife-camera-data/config.yaml:
    web:
      port: 80
      open_browser: false
    schedule:
      enabled: true     # stop the camera at night; web/gallery stays up 24/7
  then:  sudo systemctl restart crittercam-web.service crittercam-schedule.service

Verify:
    systemctl status crittercam-tracker crittercam-web
    curl http://localhost/api/status         # (port 80 once config is set)
    curl http://$(hostname).local/api/status  # from a phone on the same wifi
    systemctl list-timers 'crittercam-*'
    deploy/scripts/schedule.py print          # today's sun times + camera state

Optional (cosmetic, for readable logs): sudo timedatectl set-timezone America/Toronto
EOF
