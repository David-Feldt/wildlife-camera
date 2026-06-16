#!/bin/sh
# Watchdog (runs as root via crittercam-healthcheck.timer).
# Web is up 24/7 (gallery is always browsable) so it's always policed; the
# tracker is intentionally stopped at night, so it's only policed when the
# schedule says the camera should be running.
set -eu

REPO=/home/sajeel/wildlife-camera
PY="$REPO/.venv/bin/python"
SCHED="$REPO/deploy/scripts/schedule.py"

STATUS="$(curl -fsS --max-time 5 http://localhost/api/status 2>/dev/null || true)"
if [ -z "$STATUS" ]; then
    logger -t crittercam-health "web /api/status unreachable -> restart web"
    systemctl restart crittercam-web.service
    exit 0
fi

# Only police the tracker when the camera is supposed to be on.
"$PY" "$SCHED" --camera-should-run >/dev/null 2>&1 || exit 0

ACTION="$(printf '%s' "$STATUS" | "$PY" - <<'PYEOF'
import sys, json
try:
    d = json.loads(sys.stdin.read())
except Exception:
    raise SystemExit
age = d.get("heartbeat_age_s")
fps = d.get("infer_fps")
if not d.get("tracker_alive") or age is None or age > 15 or fps is None or fps <= 0:
    print("tracker")
PYEOF
)"

if [ "$ACTION" = "tracker" ]; then
    logger -t crittercam-health "tracker unhealthy (camera should be on) -> restart"
    systemctl restart crittercam-tracker.service
fi
