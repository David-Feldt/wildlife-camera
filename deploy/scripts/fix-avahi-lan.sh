#!/bin/sh
# Pin avahi/mDNS to the real LAN interface so <hostname>.local advertises the
# WiFi IPv4 (e.g. 10.0.0.197) — not the docker bridge (172.17.0.1) or a global
# IPv6 the IPv4-only web server can't answer. Run once: sudo deploy/scripts/fix-avahi-lan.sh
set -eu

CONF=/etc/avahi/avahi-daemon.conf

# The interface that carries the default route = the LAN/WiFi link.
IFACE="$(ip route show default 2>/dev/null | awk '{print $5; exit}')"
[ -n "$IFACE" ] || { echo "could not determine the LAN interface (no default route)"; exit 1; }

# Idempotent: drop any keys we manage, then re-add them under the right sections.
sed -i '/^allow-interfaces=/d; /^publish-aaaa-on-ipv4=/d; /^use-ipv6=/d' "$CONF"
sed -i "/^\[server\]/a allow-interfaces=$IFACE\nuse-ipv6=no" "$CONF"
sed -i "/^\[publish\]/a publish-aaaa-on-ipv4=no" "$CONF"

systemctl restart avahi-daemon
sleep 1

echo "avahi pinned to '$IFACE'; $(hostname).local now resolves to:"
getent ahostsv4 "$(hostname).local" | awk '{print "  " $1}' | sort -u
