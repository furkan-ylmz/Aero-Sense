# Aero-Sense: Real-Time Tactical Telemetry and Maneuver Classification System

[![CI/CD Pipeline](https://github.com/furkan-ylmz/Aero-Sense/actions/workflows/ci.yml/badge.svg)](https://github.com/furkan-ylmz/Aero-Sense/actions)
![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Cross--Platform%20(Linux%20%26%20Windows)-green.svg)
![Latency](https://img.shields.io/badge/Inference%20Latency-%3C%3D%202.0%20ms-brightgreen.svg)

---

## 1. Executive Summary and Scope

**Aero-Sense** is a low-latency, defense-grade tactical command and control (C2) situational awareness subsystem. It processes high-frequency 3D radar and transponder (ADS-B / IFF) telemetry streams in real time, extracts dynamic kinematic features, and classifies target flight maneuvers using a hybrid deep learning model (1D-CNN + Bidirectional LSTM).

The inference core is implemented in cross-platform **C++17** with CPU SIMD/AVX2 acceleration and strict zero-dynamic-allocation design patterns, delivering deterministic inference latencies well below the **<= 2.0 ms** operational defense threshold.

---

## 2. System Architecture and Pipeline

```
[Radar / OpenSky ADS-B Streamer]
               │ (Raw UDP: Port 5005)
               ▼
[C++17 Core Engine: Multi-Track State Management]
  ├── Thread-Safe Circular Ring Buffer (30x8 Sliding Window)
  ├── Kinematic Derivative & Z-Score Normalization
  └── Optimized Inference Engine (1D-CNN + BiLSTM)
               │ (Binary TelemetryPacket UDP: Port 5006)
               ▼
[Telemetry Forwarder Bridge] ──> [InfluxDB 2.7 Time-Series DB]
                                           │ (Flux Queries)
                                           ▼
                       [Grafana Tactical C2 Dashboard & Live GeoMap]
```

---

## 3. Kinematic State Model (8-Dimensional Vector)

The classification pipeline samples continuous flight trajectories at 1 Hz, maintaining a rolling temporal window of 30 seconds ($30 \times 8$ matrix):

| Index | Symbol | Parameter Name | Unit | Mathematical Definition |
|:---:|:---:|---|:---:|---|
| 0 | $x$ | East Relative Position | m | Cartesian East relative to radar / origin |
| 1 | $y$ | North Relative Position | m | Cartesian North relative to radar / origin |
| 2 | $z$ | Altitude | m | Barometric or geometric altitude |
| 3 | $v_g$ | Ground Speed | m/s | $v_g = \sqrt{\dot{x}^2 + \dot{y}^2}$ |
| 4 | $\theta$ | Heading / Track Angle | deg | $\theta = \text{atan2}(\dot{y}, \dot{x}) \pmod{360}$ |
| 5 | $\omega$ | Yaw / Turn Rate | deg/s | $\omega_t = (\theta_t - \theta_{t-1}) / \Delta t$ (Normalized to $[-180^\circ, +180^\circ]$) |
| 6 | $a_t$ | Tangential Acceleration | $\text{m/s}^2$ | $a_{t} = (v_{g,t} - v_{g,t-1}) / \Delta t$ |
| 7 | $a_z$ | Vertical Acceleration | $\text{m/s}^2$ | $a_{z,t} = (v_{z,t} - v_{z,t-1}) / \Delta t$ |

---

## 4. Tactical Maneuver Taxonomy

The classifier identifies 6 distinct operational maneuver classes:

| ID | Class Name | Kinematic Signatures | Operational Context |
|:---:|---|---|---|
| 0 | Straight Cruise | $\omega \approx 0^\circ/\text{s}$, $a_t \approx 0$, $a_z \approx 0$, constant speed/alt | Routine transit / cruise flight |
| 1 | Coordinated Turn | $\|\omega\| \in [1.5^\circ/\text{s}, 4.0^\circ/\text{s}]$, $v_z \approx 0$ | Standard heading change |
| 2 | Climb | $v_z > +5.0\,\text{m/s}$, $a_z > 0$, $\omega \approx 0$ | Altitude acquisition |
| 3 | Descent / Dive | $v_z < -5.0\,\text{m/s}$, $a_z < 0$, $\omega \approx 0$ | Tactical descent / approach |
| 4 | Orbit / Holding | Sustained continuous $\omega$, closed $360^\circ$ circle | Airspace loitering / patrol |
| 5 | Evasive Maneuver | $\|\omega\| > 6.0^\circ/\text{s}$, $\|a_t\| > 3.0\,\text{m/s}^2$, high G-load | Threat avoidance / jinking |

---

## 5. Technology Stack

* **Machine Learning:** PyTorch 2.x, ONNX Runtime 1.29, ONNX (Opset 17)
* **Core Inference:** C++17, CMake 3.16+, MSVC (Windows) / GCC (Linux), AVX2 Optimization
* **Serialization & Networking:** Google Protocol Buffers v3, Raw Cross-Platform UDP Sockets (Winsock2 / POSIX)
* **Observability & C2 Panel:** InfluxDB 2.7, Grafana OSS, Docker Compose
* **CI/CD Pipeline:** GitHub Actions (`ubuntu-22.04` and `windows-latest` dual runner matrix)

---

## 6. Installation and Execution Guide

### 6.1 Prerequisites
* Python 3.10+
* CMake 3.16+ and C++17 compliant compiler (MSVC on Windows, GCC/Clang on Linux)
* Docker & Docker Compose (for InfluxDB and Grafana)

### 6.2 Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 6.3 Step 2: Compile Protobuf Schema
```bash
python -m grpc_tools.protoc -Iproto --python_out=proto/generated proto/telemetry.proto
```

### 6.4 Step 3: Generate Dataset and Train Model
```bash
# Generate 7,200 synthetic 6-DOF trajectory windows
python scripts/generate_synthetic_data.py --samples_per_class 1200

# Train PyTorch 1D-CNN + BiLSTM and export ONNX model
python scripts/train_model.py --epochs 25
```

### 6.5 Step 4: Build C++ Inference Engine
#### Windows (Visual Studio / MSVC):
```powershell
cd src_cpp
cmake -B build
cmake --build build --config Release
```

#### Linux (Ubuntu 22.04 LTS):
```bash
cd src_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 6.6 Step 5: Start Observability Services (Docker Compose)
```bash
cd docker
docker compose up -d
```
* **Grafana Dashboard URL:** `http://localhost:3000` (Username: `admin`, Password: `admin`)
* **InfluxDB Dashboard URL:** `http://localhost:8086`

### 6.7 Step 6: Launch Real-Time Pipeline
Open two terminals:

**Terminal 1: C++ Core Inference Engine**
```powershell
# Windows
.\src_cpp\build\Release\maneuver_inference_engine.exe 5005 5006

# Linux
./src_cpp/build/maneuver_inference_engine 5005 5006
```

**Terminal 2: Telemetry Streamer (OpenSky ADS-B / Synthetic)**
```powershell
python scripts/opensky_streamer.py --host 127.0.0.1 --port 5005
```

---

## 7. Performance Benchmarks and Verification

Automated integration test results (`scripts/test_end_to_end.py`):

| Test Scenario | Ground Truth Class | Predicted Class | Model Confidence | Core Latency | Status |
|---|---|---|:---:|:---:|:---:|
| `TRK-CRUISE-01` | Straight Cruise | `STRAIGHT_CRUISE` | 91.6% | 0.005 ms | PASS |
| `TRK-TURN-02` | Coordinated Turn | `COORDINATED_TURN` | 99.3% | 0.007 ms | PASS |
| `TRK-CLIMB-03` | Climb | `CLIMB` | 100.0% | 0.004 ms | PASS |
| `TRK-EVASIVE-04` | Evasive Maneuver | `EVASIVE_MANEUVER` | 100.0% | 0.007 ms | PASS |

* **Average Core Latency:** 0.006 ms (6 microseconds)
* **Maximum Core Latency:** 0.007 ms
* **Defense SLA Requirement:** $\le 2.0\text{ ms}$ (Passed with $> 250\times$ margin)
* **Numerical Parity (PyTorch vs. ONNX Runtime):** $MSE = 3.45 \times 10^{-13}$

---

## 8. Repository Layout

```
Aero-Sense/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Cross-platform GitHub Actions CI workflow
├── data/
│   ├── raw/                         # Raw ADS-B and radar logs
│   └── processed/                   # Preprocessed .npy datasets and norm_params.json
├── docker/
│   ├── docker-compose.yml           # InfluxDB 2.7, Grafana, and Telemetry Forwarder
│   ├── Dockerfile.forwarder         # Forwarder container specification
│   ├── telemetry_forwarder.py       # UDP 5006 to InfluxDB Line Protocol service
│   └── grafana/provisioning/        # Automated datasource and dashboard configs
├── models/
│   ├── model_best.pt                # Trained PyTorch model checkpoint
│   └── model_cnn_lstm.onnx          # Production ONNX model
├── proto/
│   ├── telemetry.proto              # Google Protocol Buffers v3 schema
│   └── generated/                   # Generated Python Protobuf bindings
├── scripts/
│   ├── generate_synthetic_data.py   # 6-class 6DOF kinematic flight generator
│   ├── train_model.py               # Deep learning training and ONNX export
│   ├── opensky_streamer.py          # Real-time ADS-B broadcaster with dead-reckoning
│   └── test_end_to_end.py           # Automated integration benchmark
├── src_cpp/
│   ├── CMakeLists.txt               # Cross-platform CMake build configuration
│   ├── include/
│   │   ├── ring_buffer.hpp          # Zero-allocation thread-safe circular buffer
│   │   ├── onnx_engine.hpp          # High-performance inference engine wrapper
│   │   └── udp_socket.hpp           # Cross-platform raw UDP socket abstraction
│   └── src/
│       └── main.cpp                 # Multi-target real-time inference loop
├── Aero-Sense.md                    # System Architecture & Technical Specification
├── README.md                        # Quick Start & Operational Documentation
└── requirements.txt                 # Python dependencies
```

---

## 9. License and Attribution

This project is developed under open-source defense R&D standards for real-time radar telemetry processing and tactical situational awareness research.
