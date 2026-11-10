#!/bin/bash
set -euo pipefail

# The evaluation host resolves localhost to IPv6, while XPolicyLab's upstream
# helper probes only an IPv4 wildcard ephemeral port. Probe both loopback
# families so the selected WebSocket port is actually bindable by the server.
FREE_PORT=$(python3 - <<'PY'
import socket

for port in range(20000, 30000):
    sockets = []
    try:
        ipv4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ipv4.bind(("127.0.0.1", port))
        sockets.append(ipv4)

        ipv6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        ipv6.bind(("::1", port))
        sockets.append(ipv6)
    except OSError:
        for candidate in sockets:
            candidate.close()
        continue
    for candidate in sockets:
        candidate.close()
    print(port)
    break
else:
    raise RuntimeError("no free dual-stack loopback port in 20000:30000")
PY
)

echo -e "\033[33m[INFO] Using dual-stack loopback port: ${FREE_PORT}\033[0m" >&2
echo "${FREE_PORT}"
