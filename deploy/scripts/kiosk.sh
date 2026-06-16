#!/bin/sh
# Kiosk launcher — started by the GNOME autologin session via
# ~/.config/autostart/crittercam-kiosk.desktop. Disables screen blanking,
# hides the cursor, and keeps a full-screen browser pointed at the local UI,
# relaunching it if it ever exits.
export DISPLAY="${DISPLAY:-:0}"
URL="http://localhost/"

# X-level blanking / power-save off.
xset s off || true
xset -dpms || true
xset s noblank || true

# Hide the pointer when idle (unclutter installed by install.sh).
command -v unclutter >/dev/null 2>&1 && unclutter -idle 1 -root &

# GNOME-level idle/lock off (these are what actually re-arm blanking under gnome-shell).
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true

# Wait for the web server to answer before the first launch (it may still be
# binding :80 right after boot).
i=0
while [ "$i" -lt 60 ]; do
    curl -fsS --max-time 2 "$URL" >/dev/null 2>&1 && break
    i=$((i + 1))
    sleep 1
done

CHROME="$(command -v chromium-browser || command -v chromium || true)"

while true; do
    if [ -n "$CHROME" ]; then
        "$CHROME" --kiosk --noerrdialogs --disable-infobars \
            --disable-session-crashed-bubble --disable-features=Translate \
            --incognito --check-for-update-interval=31536000 "$URL" || true
    else
        # Fallback: Epiphany web-app mode (already installed).
        epiphany -a --profile="$HOME/.local/share/crittercam-kiosk" "$URL" || true
    fi
    sleep 3   # browser closed/crashed — relaunch
done
