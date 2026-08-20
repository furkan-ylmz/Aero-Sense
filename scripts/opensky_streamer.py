#!/usr/bin/env python3
"""
Aero-Sense: OpenSky Network ADS-B Telemetry Streamer & UDP Publisher
Fetches real aircraft telemetry from OpenSky API (or generates realistic flight streams if offline),
encodes data into Protobuf / UDP packets, and streams to the C++ Core Inference Engine (Port 5005).
"""

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.request

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr:
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


OPENSKY_URL = "https://opensky-network.org/api/states/all"


def fetch_opensky_states(bounding_box: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    Fetches live aircraft state vectors from OpenSky Network.
    bounding_box: [lamin, lomin, lamax, lomax] (Optional)
    """
    url = OPENSKY_URL
    if bounding_box and len(bounding_box) == 4:
        url += f"?lamin={bounding_box[0]}&lomin={bounding_box[1]}&lamax={bounding_box[2]}&lomax={bounding_box[3]}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Aero-Sense-Defense-Research/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            states = data.get("states", [])
            parsed_aircraft = []
            for s in states:
                # OpenSky format: [0: icao24, 1: callsign, 2: origin_country, 3: time_position,
                #                  4: last_contact, 5: longitude, 6: latitude, 7: baro_altitude,
                #                  8: on_ground, 9: velocity, 10: true_track, 11: vertical_rate]
                if s[5] is not None and s[6] is not None and not s[8]:
                    parsed_aircraft.append({
                        "icao24": s[0],
                        "callsign": (s[1] or s[0]).strip(),
                        "lon": float(s[5]),
                        "lat": float(s[6]),
                        "alt": float(s[7] or 0.0),
                        "vel": float(s[9] or 0.0),
                        "track": float(s[10] or 0.0),
                        "vertical_rate": float(s[11] or 0.0),
                        "timestamp": int(time.time() * 1000),
                    })
            return parsed_aircraft
    except Exception as e:
        print(f"[!] OpenSky API query notice ({e}). Falling back to internal trajectory generator.")
        return []


def pack_telemetry_binary(
    track_id: str,
    timestamp_ms: int,
    lat: float,
    lon: float,
    alt_m: float,
    speed_mps: float,
    track_deg: float,
) -> bytes:
    """
    Packs telemetry point into binary buffer for raw UDP transport.
    Struct format: 16s Q d d d d d (TrackID: 16 bytes, Timestamp: uint64, Lat/Lon/Alt/Speed/Heading: double)
    Total size: 16 + 8 + 5*8 = 64 bytes fixed size.
    """
    track_id_bytes = track_id.encode("utf-8")[:16].ljust(16, b"\x00")
    return struct.pack(
        "<16sQddddd",
        track_id_bytes,
        timestamp_ms,
        lat,
        lon,
        alt_m,
        speed_mps,
        track_deg,
    )


def run_streamer(
    host: str = "127.0.0.1",
    port: int = 5005,
    rate_hz: float = 1.0,
    bbox: Optional[List[float]] = None,
    poll_interval_sec: float = 10.0,
) -> None:
    """
    Main loop streaming live aircraft telemetry with Kinematic Dead-Reckoning Extrapolation.
    Polls OpenSky API every 10 seconds to respect anonymous rate limits, and extrapolates flight vectors at 1 Hz.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[*] Aero-Sense Telemetry Streamer active -> Broadcasting to UDP {host}:{port} @ {rate_hz} Hz")
    print(f"[*] OpenSky Polling Interval: {poll_interval_sec}s (with 1 Hz Kinematic Dead-Reckoning)")

    sim_aircraft = [
        {"icao24": "TURKISH-AF-01", "callsign": "THK-01", "lat": 39.9334, "lon": 32.8597, "alt": 7500.0, "vel": 240.0, "track": 45.0, "vertical_rate": 2.0},
        {"icao24": "NATO-PATROL-42", "callsign": "NATO-42", "lat": 40.1500, "lon": 33.1000, "alt": 9200.0, "vel": 220.0, "track": 180.0, "vertical_rate": -1.0},
    ]

    cached_aircraft = []
    last_poll_time = 0.0
    seq = 0

    try:
        while True:
            t0 = time.time()

            # Poll OpenSky Network API periodically (every 10s)
            if time.time() - last_poll_time >= poll_interval_sec:
                fresh_states = fetch_opensky_states(bbox)
                if fresh_states:
                    cached_aircraft = fresh_states
                    print(f"[*] Refreshed live flight radar ({len(cached_aircraft)} active targets in airspace)")
                elif not cached_aircraft:
                    cached_aircraft = sim_aircraft
                last_poll_time = time.time()

            # Kinematic Dead-Reckoning (Propagate position at 1 Hz)
            dt = 1.0 / rate_hz
            for ac in cached_aircraft:
                rad = math_radians(ac.get("track", 0.0))
                # 1 degree latitude ~ 111,000 meters
                lat_rad = math_radians(ac.get("lat", 0.0))
                cos_lat = max(0.1, math_cos(lat_rad))
                ac["lat"] += (ac.get("vel", 0.0) * math_cos(rad) * dt) / 111000.0
                ac["lon"] += (ac.get("vel", 0.0) * math_sin(rad) * dt) / (111000.0 * cos_lat)
                ac["alt"] += ac.get("vertical_rate", 0.0) * dt
                ac["timestamp"] = int(time.time() * 1000)

            # Stream up to 5 aircraft to UDP socket
            for ac in cached_aircraft[:5]:
                packet_bytes = pack_telemetry_binary(
                    track_id=ac.get("callsign", ac.get("icao24", "TRK-01")),
                    timestamp_ms=ac.get("timestamp", int(time.time() * 1000)),
                    lat=float(ac.get("lat", 0.0)),
                    lon=float(ac.get("lon", 0.0)),
                    alt_m=float(ac.get("alt", ac.get("altitude", 0.0))),
                    speed_mps=float(ac.get("vel", ac.get("velocity", 0.0))),
                    track_deg=float(ac.get("track", ac.get("heading", 0.0))),
                )
                sock.sendto(packet_bytes, (host, port))
                seq += 1

            if seq % 10 == 0:
                print(f"[+] Broadcasted {seq} telemetry frames (Active targets: {len(cached_aircraft[:5])})")

            elapsed = time.time() - t0
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[*] Telemetry streamer stopped by user.")
    finally:
        sock.close()


def math_radians(deg: float) -> float:
    import math
    return math.radians(deg)


def math_cos(rad: float) -> float:
    import math
    return math.cos(rad)


def math_sin(rad: float) -> float:
    import math
    return math.sin(rad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aero-Sense OpenSky ADS-B Telemetry Streamer")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Target UDP host")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP port")
    parser.add_argument("--rate", type=float, default=1.0, help="Publish rate in Hz")
    parser.add_argument("--bbox", nargs=4, type=float, default=[36.0, 26.0, 42.0, 45.0], help="Turkey & Aegean bounding box: lamin lomin lamax lomax")
    args = parser.parse_args()

    run_streamer(host=args.host, port=args.port, rate_hz=args.rate, bbox=args.bbox)
