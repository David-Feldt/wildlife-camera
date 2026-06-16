#!/usr/bin/env python3
"""Day/night camera scheduler for the crittercam appliance.

Computes sunrise/sunset for the configured lat/lon (no third-party deps — a
self-contained NOAA/Almanac solar calc) and switches the camera with the sun:
the tracker (capture + inference + recording) is stopped at sunset and started
again at sunrise. The box stays powered and the web UI keeps running, so the
gallery is browsable around the clock. All times are absolute UTC epoch, so the
system timezone is irrelevant to correctness.

Subcommands:
  apply                 set the camera to the correct state now and schedule the
                        next sunrise/sunset transition (re-arms a transient unit)
  print                 print today's / upcoming sun times (UTC + local) and exit
  --is-daytime          exit 0 if the sun is currently up, 1 if down
  --camera-should-run   exit 0 if the camera should be running now (daytime, or
                        the schedule is disabled), 1 if it should be off

Run as root (it shells out to systemctl / systemd-run for the system manager).
"""
from __future__ import annotations

import datetime as dt
import math
import subprocess
import sys
from pathlib import Path

# crittercam.config is light (yaml + pydantic only — no torch/ultralytics).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crittercam.config import load_config  # noqa: E402

SELF = str(Path(__file__).resolve())
TRACKER_UNIT = "crittercam-tracker.service"
DAYNIGHT_UNIT = "crittercam-daynight"   # transient unit that re-runs `apply`
ZENITH = 90.833  # official sunrise/sunset (sun's upper limb + refraction)
UTC = dt.timezone.utc
DEVNULL = subprocess.DEVNULL


def _sun_event_epoch(date: dt.date, lat: float, lon: float, rising: bool) -> float | None:
    """Sunrise (rising=True) or sunset epoch for a calendar date, or None on a
    polar day where the event does not occur. Almanac for Computers algorithm."""
    n = date.toordinal() - dt.date(date.year, 1, 1).toordinal() + 1
    lng_hour = lon / 15.0
    t = n + ((6 if rising else 18) - lng_hour) / 24.0
    m = 0.9856 * t - 3.289
    L = (m + 1.916 * math.sin(math.radians(m))
         + 0.020 * math.sin(math.radians(2 * m)) + 282.634) % 360
    ra = math.degrees(math.atan(0.91764 * math.tan(math.radians(L)))) % 360
    # bring RA into the same quadrant as L
    ra += (math.floor(L / 90) * 90) - (math.floor(ra / 90) * 90)
    ra /= 15.0
    sin_dec = 0.39782 * math.sin(math.radians(L))
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = ((math.cos(math.radians(ZENITH)) - sin_dec * math.sin(math.radians(lat)))
             / (cos_dec * math.cos(math.radians(lat))))
    if cos_h > 1 or cos_h < -1:
        return None  # sun never rises / never sets on this date at this latitude
    h = (360 - math.degrees(math.acos(cos_h))) if rising else math.degrees(math.acos(cos_h))
    h /= 15.0
    local_mean = h + ra - 0.06571 * t - 6.622
    ut = (local_mean - lng_hour) % 24
    midnight = dt.datetime(date.year, date.month, date.day, tzinfo=UTC)
    return (midnight + dt.timedelta(hours=ut)).timestamp()


def _next_event_after(ts: float, lat: float, lon: float, rising: bool,
                      margin_sec: float = 0.0) -> float | None:
    """First sun event strictly after `ts` (epoch), scanning nearby UTC dates so
    we never mis-attribute a sunset that falls past UTC midnight."""
    base = dt.datetime.fromtimestamp(ts, tz=UTC).date()
    for delta in (-1, 0, 1, 2):
        e = _sun_event_epoch(base + dt.timedelta(days=delta), lat, lon, rising)
        if e is None:
            continue
        e += margin_sec
        if e > ts:
            return e
    return None


def _windows(now: float, cfg):
    """Return (next_camera_off, next_camera_on, is_daytime) including margins:
    camera goes off `after_sunset` min past sunset, on `before_sunrise` min
    before sunrise."""
    lat, lon = cfg.schedule.latitude, cfg.schedule.longitude
    after = cfg.schedule.margin_after_sunset_min * 60
    before = cfg.schedule.margin_before_sunrise_min * 60
    off_at = _next_event_after(now, lat, lon, rising=False, margin_sec=after)   # sunset
    on_at = _next_event_after(now, lat, lon, rising=True, margin_sec=-before)   # sunrise
    daytime = off_at is not None and on_at is not None and off_at < on_at
    return off_at, on_at, daytime


def _fmt(ts: float | None) -> str:
    if ts is None:
        return "none (polar day/night)"
    u = dt.datetime.fromtimestamp(ts, tz=UTC)
    loc = dt.datetime.fromtimestamp(ts).astimezone()
    return f"{u:%Y-%m-%d %H:%M} UTC / {loc:%Y-%m-%d %H:%M %Z}"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _systemctl(*args: str) -> None:
    subprocess.run(["systemctl", *args], check=False)


def cmd_apply(cfg) -> int:
    # Clear any previously-armed transition so re-running is idempotent.
    _systemctl("stop", f"{DAYNIGHT_UNIT}.timer")
    _systemctl("reset-failed", f"{DAYNIGHT_UNIT}.timer", f"{DAYNIGHT_UNIT}.service")

    if not cfg.schedule.enabled:
        print("schedule disabled — tracker runs 24/7, no day/night switching")
        return 0

    now = dt.datetime.now(tz=UTC).timestamp()
    off_at, on_at, daytime = _windows(now, cfg)
    if daytime:
        print(f"daytime — camera ON; next off (sunset) {_fmt(off_at)}")
        _systemctl("start", TRACKER_UNIT)
        nxt = off_at
    else:
        print(f"night — camera OFF; next on (sunrise) {_fmt(on_at)}")
        _systemctl("stop", TRACKER_UNIT)
        nxt = on_at

    if nxt is None:
        print("no next transition (polar) — leaving camera state as is", file=sys.stderr)
        return 0

    delay = max(int(nxt - now), 30)
    print(f"next transition in {delay}s")
    _run(["systemd-run", f"--unit={DAYNIGHT_UNIT}", "--collect",
          f"--on-active={delay}s", "--timer-property=AccuracySec=30s",
          sys.executable, SELF, "apply"])
    return 0


def main() -> int:
    cfg = load_config()
    args = sys.argv[1:]
    now = dt.datetime.now(tz=UTC).timestamp()

    if "--is-daytime" in args:
        return 0 if _windows(now, cfg)[2] else 1
    if "--camera-should-run" in args:
        if not cfg.schedule.enabled:
            return 0  # 24/7 operation
        return 0 if _windows(now, cfg)[2] else 1
    if args and args[0] == "print":
        off_at, on_at, daytime = _windows(now, cfg)
        print(f"lat={cfg.schedule.latitude} lon={cfg.schedule.longitude} "
              f"enabled={cfg.schedule.enabled} mode={cfg.schedule.mode}")
        print(f"now            = {_fmt(now)}")
        print(f"next sunset    = {_fmt(off_at)}  (camera off)")
        print(f"next sunrise   = {_fmt(on_at)}  (camera on)")
        print(f"currently      = {'DAY (camera on)' if daytime else 'NIGHT (camera off)'}")
        return 0
    if args and args[0] == "apply":
        return cmd_apply(cfg)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
