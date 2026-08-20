#!/usr/bin/env python3
"""
Aero-Sense: Telemetry Forwarder & InfluxDB Bridge
Listens for classified telemetry packets on UDP port 5006, formats metrics into
InfluxDB Line Protocol, and pushes records into the time-series database.
"""

import json
import os
import socket
import struct
import time
import urllib.request
import urllib.error

INFLUX_URL = os.getenv("INFLUX_URL", "http://127.0.0.1:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "telemetry-secret-token-2026")
INFLUX_ORG = os.getenv("INFLUX_ORG", "defense_org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "telemetry_stream")
UDP_PORT = int(os.getenv("UDP_PORT", "5006"))

WRITE_API_URL = f"{INFLUX_URL}/api/v2/write?org={INFLUX_ORG}&bucket={INFLUX_BUCKET}&precision=ms"

CLASS_MAP = {
    0: "CRUISE",
    1: "COORDINATED_TURN",
    2: "CLIMB",
    3: "DESCENT",
    4: "HOLDING",
    5: "EVASIVE",
}


def send_to_influx(line_protocol: str) -> bool:
    """Sends line protocol metric to InfluxDB v2 write API."""
    req = urllib.request.Request(
        WRITE_API_URL,
        data=line_protocol.encode("utf-8"),
        headers={
            "Authorization": f"Token {INFLUX_TOKEN}",
            "Content-Type": "text/plain; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status in (200, 204)
    except urllib.error.URLError as e:
        # Silently retry on connection errors if InfluxDB is initializing
        return False


def run_forwarder() -> None:
    """Main UDP receiver and InfluxDB writer loop."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"[*] Telemetry Forwarder listening on UDP 0.0.0.0:{UDP_PORT}")
    print(f"[*] Forwarding target -> {WRITE_API_URL}")

    count = 0
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            if not data:
                continue

            # Check if JSON payload or Binary
            if data.startswith(b"{"):
                msg = json.loads(data.decode("utf-8"))
                track_id = msg.get("track_id", "TRK-UNKNOWN")
                ts = msg.get("timestamp", int(time.time() * 1000))
                lat = float(msg.get("lat", 0.0))
                lon = float(msg.get("lon", 0.0))
                alt = float(msg.get("alt", 0.0))
                speed = float(msg.get("speed", 0.0))
                heading = float(msg.get("heading", 0.0))
                maneuver_id = int(msg.get("maneuver_id", 0))
                confidence = float(msg.get("confidence", 1.0))
                latency_ms = float(msg.get("latency_ms", 1.0))
            else:
                # Binary format: 16s Q d d d d d i f f (64 + 4 + 4 + 4 = 76 bytes)
                try:
                    track_raw, ts, lat, lon, alt, speed, heading, maneuver_id, confidence, latency_ms = struct.unpack(
                        "<16sQdddddiff", data[:76]
                    )
                    track_id = track_raw.decode("utf-8", errors="ignore").rstrip("\x00")
                except Exception:
                    continue

            maneuver_name = CLASS_MAP.get(maneuver_id, f"CLASS_{maneuver_id}")

            # Construct InfluxDB Line Protocol
            # Format: flight_telemetry,track_id=TRK-01,maneuver=CRUISE lat=39.9,lon=32.8,alt=7500.0,speed=240.0,heading=45.0,confidence=0.98,latency_ms=1.42 1692440000000
            line = (
                f"flight_telemetry,track_id={track_id},maneuver={maneuver_name} "
                f"latitude={lat},longitude={lon},altitude_m={alt},ground_speed_mps={speed},"
                f"track_angle_deg={heading},confidence={confidence},latency_ms={latency_ms},maneuver_id={maneuver_id}i "
                f"{ts}"
            )

            send_to_influx(line)
            count += 1
            if count % 20 == 0:
                print(f"[+] Ingested {count} telemetry records -> [{track_id}] {maneuver_name} (Lat: {lat:.3f}, Lon: {lon:.3f}, Latency: {latency_ms:.2f}ms)")

        except Exception as e:
            print(f"[!] Forwarder error: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    run_forwarder()
