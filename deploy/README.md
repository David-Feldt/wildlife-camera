# deploy/ — unattended appliance setup

Turns a bench install into an auto-starting kiosk that runs the camera only in
daylight, reachable at `http://<hostname>.local`. See the project CLAUDE.md for
the wider system.

## Install

```bash
sudo deploy/install.sh
```

Then set the runtime overrides in `~/wildlife-camera-data/config.yaml`:

```yaml
web:
  port: 80           # clean address; the web unit has CAP_NET_BIND_SERVICE
  open_browser: false # the kiosk owns the browser now
schedule:
  enabled: true      # arm the sunset-suspend / sunrise-wake cycle
```

and `sudo systemctl restart crittercam-web crittercam-schedule`.

## What gets installed

| Unit | Role |
|------|------|
| `crittercam-tracker.service` | capture + inference + recording (camera, GPU groups) |
| `crittercam-web.service` | FastAPI UI, binds **:80** via `CAP_NET_BIND_SERVICE` |
| `crittercam-schedule.{service,timer}` | boot + daily 12:05 → set camera state, arm next transition |
| `~/.config/autostart/crittercam-kiosk.desktop` | launches `kiosk.sh` (Chromium `--kiosk`) |

## Day/night cycle (camera-off, box stays on)

`schedule.py` computes sunrise/sunset for `schedule.latitude/longitude` (default
Toronto/GTA) with a self-contained NOAA calc — no third-party deps, all in UTC
epoch so the system timezone doesn't affect correctness. `schedule.py apply`
stops `crittercam-tracker` at sunset and starts it at sunrise, then arms a
transient `crittercam-daynight` unit to re-run itself at the next transition. The
box stays powered and `crittercam-web` runs 24/7, so the gallery is always
browsable — only the camera sleeps. `enabled: false` runs the tracker 24/7.

```bash
deploy/scripts/schedule.py print              # today's sun times + camera state
deploy/scripts/schedule.py --is-daytime       # exit 0 day / 1 night
deploy/scripts/schedule.py --camera-should-run # exit 0 if the camera should be on now
```

## Privileges

No standing `sudoers` rule: the schedule/healthcheck paths run as **root system
units**, so the only privileged step is the one-time `sudo deploy/install.sh`.

## mDNS / LAN name

`fix-avahi-lan.sh` pins avahi to the WiFi interface and disables IPv6 publishing,
so `<hostname>.local` resolves to the LAN IPv4 (not the docker bridge or a global
IPv6 the IPv4-only server can't answer). Run once with sudo if the `.local` name
points at the wrong address.

## Bench validation (before mounting outdoors)

```bash
systemctl status crittercam-tracker crittercam-web   # both active in daylight
deploy/scripts/schedule.py print                     # sanity-check sun times
# Force a transition test: temporarily set a far-south/near latitude or check
# that stopping/starting works:  sudo systemctl start crittercam-schedule.service
```

The one thing to confirm on hardware: the Flathub Chromium
(`flatpak run org.chromium.Chromium`) renders the live UI in `--kiosk` (else
`kiosk.sh` falls back to Epiphany). The distro `chromium-browser` snap does **not**
work on this Tegra kernel — snap-confine aborts with `cap_dac_override not found`
(no AppArmor), so we run Chromium from Flathub under bubblewrap instead. It needs
`--disable-gpu` because the nvidia EGL stack isn't visible inside the sandbox;
Chromium falls back to software rendering, which is fine for the MJPEG UI.
