#!/usr/bin/env python3
"""
Aero-Sense: End-to-End System Integration Test
Launches C++ Core Inference Engine, streams synthetic tactical maneuvers over UDP:5005,
and verifies real-time maneuver classification output & latency (< 2.0 ms) on UDP:5006.
"""

import math
import os
import socket
import struct
import subprocess
import sys
import time
from typing import Dict, List, Tuple

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CPP_EXECUTABLE = os.path.join("src_cpp", "build", "Release", "maneuver_inference_engine.exe")
if not os.path.exists(CPP_EXECUTABLE):
    # Linux fallback
    CPP_EXECUTABLE = os.path.join("src_cpp", "build", "maneuver_inference_engine")

INPUT_PORT = 5005
OUTPUT_PORT = 5006

CLASS_NAMES = [
    "STRAIGHT_CRUISE",
    "COORDINATED_TURN",
    "CLIMB",
    "DESCENT_DIVE",
    "ORBIT_HOLDING",
    "EVASIVE_MANEUVER",
]


def pack_telemetry_binary(track_id: str, ts: int, lat: float, lon: float, alt: float, speed: float, heading: float) -> bytes:
    track_id_bytes = track_id.encode("utf-8")[:16].ljust(16, b"\x00")
    return struct.pack("<16sQddddd", track_id_bytes, ts, lat, lon, alt, speed, heading)


def run_integration_test() -> bool:
    print("=" * 70)
    print("  AERO-SENSE: END-TO-END SYSTEM INTEGRATION & LATENCY BENCHMARK")
    print("=" * 70)

    if not os.path.exists(CPP_EXECUTABLE):
        print(f"[!] Executable not found at: {CPP_EXECUTABLE}")
        return False

    print(f"[*] Starting C++ Core Engine: {CPP_EXECUTABLE}...")
    cpp_proc = subprocess.Popen(
        [CPP_EXECUTABLE, str(INPUT_PORT), str(OUTPUT_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.0)  # Allow socket bind

    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind(("0.0.0.0", OUTPUT_PORT))
    rx_sock.settimeout(3.0)

    test_scenarios = [
        ("TRK-CRUISE-01", 0, "STRAIGHT_CRUISE", 7000.0, 230.0, 45.0, 0.0, 0.0),
        ("TRK-TURN-02", 1, "COORDINATED_TURN", 7000.0, 220.0, 90.0, 2.5, 0.0),
        ("TRK-CLIMB-03", 2, "CLIMB", 8500.0, 210.0, 90.0, 0.0, 15.0),
        ("TRK-EVASIVE-04", 5, "EVASIVE_MANEUVER", 6000.0, 290.0, 180.0, 8.0, 10.0),
    ]

    all_passed = True
    latencies = []

    try:
        for track_id, expected_cls_id, expected_name, base_alt, base_speed, base_hdg, d_hdg, d_alt in test_scenarios:
            print(f"\n[*] Streaming 32 seconds of flight for [{track_id}] (Target: {expected_name})...")
            curr_hdg = base_hdg
            curr_alt = base_alt
            curr_speed = base_speed

            # Stream 32 samples to fill the 30-sample ring buffer
            for t in range(32):
                curr_hdg = (curr_hdg + d_hdg) % 360.0
                curr_alt += d_alt
                if expected_cls_id == 5:
                    # Inject high speed acceleration for evasive
                    curr_speed += (4.0 if t % 2 == 0 else -4.0)

                pkt = pack_telemetry_binary(
                    track_id=track_id,
                    ts=int(time.time() * 1000),
                    lat=39.9 + (t * 0.001),
                    lon=32.8 + (t * 0.001),
                    alt=curr_alt,
                    speed=curr_speed,
                    heading=curr_hdg,
                )
                tx_sock.sendto(pkt, ("127.0.0.1", INPUT_PORT))
                time.sleep(0.01)  # High-speed test stream

            # Receive latest classified output packet after stream
            time.sleep(0.05)
            latest_pkt = None
            rx_sock.settimeout(0.1)
            while True:
                try:
                    data, _ = rx_sock.recvfrom(1024)
                    latest_pkt = data
                except socket.timeout:
                    break

            if latest_pkt:
                track_raw, ts, lat, lon, alt, speed, hdg, detected_cls, conf, latency_ms = struct.unpack(
                    "<16sQdddddiff", latest_pkt[:76]
                )
                detected_name = CLASS_NAMES[detected_cls] if 0 <= detected_cls < 6 else f"CLASS_{detected_cls}"
                latencies.append(latency_ms)

                is_ok = (detected_cls == expected_cls_id)
                status_str = "PASS" if is_ok else "WARN"
                print(f"  -> Result: Detected [{detected_name}] | Conf: {conf*100:.1f}% | Latency: {latency_ms:.3f} ms [{status_str}]")

                if not is_ok:
                    all_passed = False
                if latency_ms > 2.0:
                    print(f"  [!] Latency warning: {latency_ms:.3f} ms exceeded 2.0 ms budget!")
            else:
                print(f"  [!] Timeout waiting for classification response on {track_id}")
                all_passed = False

    finally:
        tx_sock.close()
        rx_sock.close()
        cpp_proc.terminate()
        try:
            cpp_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            cpp_proc.kill()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    print("\n" + "=" * 70)
    print("  INTEGRATION TEST BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total Scenarios Tested:  {len(test_scenarios)}")
    print(f"  Average Core Latency:    {avg_latency:.3f} ms")
    print(f"  Maximum Core Latency:    {max_latency:.3f} ms")
    print(f"  Latency Budget Check:    {'PASSED (<= 2.0 ms)' if max_latency <= 2.0 else 'FAILED'}")
    print("=" * 70)

    return all_passed and (max_latency <= 2.0)


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
