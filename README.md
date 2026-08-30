<div align="center">

[English](#english) | [Türkçe](#türkçe)

</div>

---

<a name="english"></a>
# Real-Time Telemetry & Maneuver Classifier

[![CI/CD Pipeline](https://github.com/furkan-ylmz/Aero-Sense/actions/workflows/ci.yml/badge.svg)](https://github.com/furkan-ylmz/Aero-Sense/actions)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-green.svg)
![Latency](https://img.shields.io/badge/Latency-%3C%3D%202.0%20ms-brightgreen.svg)

Aero-Sense is a high-performance, defense-grade tactical command and control (C2) situational awareness subsystem designed for real-time 3D radar and transponder (ADS-B / IFF) telemetry processing.

The system extracts 8D kinematic flight derivatives from multi-target trajectories and classifies 6 tactical maneuvers using a hybrid 1D-CNN + BiLSTM model powered by a zero-allocation C++17 inference core.

---

## System Architecture & Core Components

The architecture consists of four primary processing modules:

1. **Kinematics & Telemetry Ingestion (Streamer & Feature Extractor):**
   - High-rate ADS-B / radar state vector ingestion with kinematic dead-reckoning extrapolation for intermittent telemetry feeds.
   - Sliding window manager maintaining a 30-sample rolling temporal window with automatic angular discontinuity correction ($360^\circ \to 0^\circ$).
   - Extraction of 8-dimensional dynamic flight features: Cartesian coordinates ($x, y, z$), ground speed ($v_g$), heading ($\theta$), turn rate ($\omega$), tangential acceleration ($a_t$), and vertical acceleration ($a_z$).

2. **AeroCore (C++17 Zero-Allocation Inference Engine):**
   - Pre-allocated circular ring buffer (`RingBuffer<30, 8>`) guaranteeing zero heap allocation (`malloc`/`new`) on the real-time execution path.
   - Multi-target state management tracking independent flight kinematics per Track ID (`std::unordered_map<std::string, TrackContext>`).
   - AVX2/SIMD-accelerated forward inference engine executing quantized ONNX models with sub-millisecond latency ($\sim 0.004\text{ ms}$).

3. **Network Layer & Protocol Serialization:**
   - Strict Google Protocol Buffers v3 binary serialization for cross-language telemetry interchange.
   - Cross-platform raw UDP socket abstraction supporting Windows (`Winsock2`) and Linux (`POSIX BSD Sockets`).
   - High-throughput asynchronous UDP bridge broadcasting classification results on Port 5006.

4. **Tactical C2 & Observability (Docker InfluxDB & Grafana):**
   - High-throughput ingestion bridge writing telemetry and maneuver classification streams to InfluxDB 2.7 via Line Protocol.
   - Real-time tactical Grafana GeoMap dashboard rendering live target positions and color-coded trajectory trails based on active maneuver classes.
   - Real-time SLA gauges monitoring inference latency, prediction confidence (%), altitude profiles, and speed vectors.

---

## Technical Stack

- **Machine Learning & Modeling:** PyTorch 2.x, ONNX Runtime, ONNX (Opset 17)
- **Core Inference Engine:** C++17, CMake 3.16+, MSVC (Windows) / GCC (Linux), AVX2 SIMD Optimization
- **Protocols & Networking:** Google Protocol Buffers v3, Raw UDP Sockets (Winsock2 / POSIX)
- **Observability & Time-Series DB:** InfluxDB 2.7, Grafana OSS, Docker Compose
- **Continuous Integration:** GitHub Actions (`ubuntu-22.04` and `windows-latest` dual runner matrix)

---

## Kinematic State Model (8-Dimensional Vector)

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

## Tactical Maneuver Taxonomy

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

## Project Structure

```
Aero-Sense/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Cross-platform GitHub Actions CI workflow
├── core/                            # C++17 Real-Time Inference Core Package
│   ├── CMakeLists.txt               # Cross-platform CMake build configuration
│   ├── include/
│   │   ├── ring_buffer.hpp          # Zero-allocation thread-safe circular buffer
│   │   ├── onnx_engine.hpp          # High-performance inference engine wrapper
│   │   └── udp_socket.hpp           # Cross-platform raw UDP socket abstraction
│   └── src/
│       └── main.cpp                 # Multi-target real-time inference loop
├── data/
│   ├── raw/                         # Raw ADS-B and radar logs
│   └── processed/                   # Preprocessed .npy datasets and norm_params.json
├── docker/
│   ├── docker-compose.yml           # InfluxDB 2.7, Grafana, and Telemetry Forwarder
│   ├── Dockerfile.forwarder         # Forwarder container specification
│   ├── telemetry_forwarder.py       # UDP 5006 to InfluxDB Line Protocol service
│   └── grafana/provisioning/        # Automated datasource and dashboard configs
├── docs/
│   └── system_architecture.md       # Detailed System Architecture & Kinematics Spec
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
├── README.md                        # Bilingual documentation
└── requirements.txt                 # Python dependencies
```

---

## Installation & Usage

### 1. Install Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Compile Protobuf Schema

Compile the Google Protocol Buffers schema to generate Python message bindings:

```bash
python -m grpc_tools.protoc -Iproto --python_out=proto/generated proto/telemetry.proto
```

### 3. Generate Synthetic Dataset & Train AI Model

Generate 6-DOF synthetic flight trajectories and train the 1D-CNN + BiLSTM neural network:

```bash
# Generate 7,200 synthetic kinematic trajectory windows
python scripts/generate_synthetic_data.py --samples_per_class 1200

# Train PyTorch model and export optimized ONNX model
python scripts/train_model.py --epochs 25
```

### 4. Build C++ Core Inference Engine

#### Windows (Visual Studio / MSVC):
```powershell
cd core
cmake -B build
cmake --build build --config Release
```

#### Linux (Ubuntu 22.04 LTS):
```bash
cd core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 5. Start Observability Services (Docker Compose)

Launch InfluxDB 2.7, Grafana, and the telemetry forwarding bridge:

```bash
cd docker
docker compose up -d
```

- **Grafana Dashboard URL:** `http://localhost:3000` (Username: `admin`, Password: `admin`)
- **InfluxDB Dashboard URL:** `http://localhost:8086`

### 6. Launch Real-Time Execution Pipeline

Open two terminal instances:

**Terminal 1: C++ Core Inference Engine**
```powershell
# Windows:
.\core\build\Release\maneuver_inference_engine.exe 5005 5006

# Linux:
./core/build/maneuver_inference_engine 5005 5006
```

**Terminal 2: Telemetry Streamer**
```powershell
python scripts/opensky_streamer.py --host 127.0.0.1 --port 5005
```

### 7. Run Automated End-to-End Benchmark Test

Validate system latency and maneuver classification accuracy against synthetic flight scenarios:

```bash
python scripts/test_end_to_end.py
```

---

## Performance Benchmarks & Verification

Automated integration test results (`scripts/test_end_to_end.py`):

| Test Scenario | Ground Truth Class | Predicted Class | Model Confidence | Core Latency | Status |
|---|---|---|:---:|:---:|:---:|
| `TRK-CRUISE-01` | Straight Cruise | `STRAIGHT_CRUISE` | 91.6% | 0.004 ms | PASS |
| `TRK-TURN-02` | Coordinated Turn | `COORDINATED_TURN` | 99.3% | 0.003 ms | PASS |
| `TRK-CLIMB-03` | Climb | `CLIMB` | 100.0% | 0.006 ms | PASS |
| `TRK-EVASIVE-04` | Evasive Maneuver | `EVASIVE_MANEUVER` | 100.0% | 0.002 ms | PASS |

- **Average Core Latency:** 0.004 ms (4 microseconds)
- **Maximum Core Latency:** 0.006 ms
- **Defense SLA Requirement:** $\le 2.0\text{ ms}$ (Passed with $> 300\times$ margin)
- **Numerical Parity (PyTorch vs. ONNX Runtime):** $MSE = 3.45 \times 10^{-13}$

<br>

---

<a name="türkçe"></a>
# Gerçek Zamanlı Telemetri ve Manevra Sınıflandırıcı

[![CI/CD Pipeline](https://github.com/furkan-ylmz/Aero-Sense/actions/workflows/ci.yml/badge.svg)](https://github.com/furkan-ylmz/Aero-Sense/actions)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-green.svg)
![Latency](https://img.shields.io/badge/Latency-%3C%3D%202.0%20ms-brightgreen.svg)

Aero-Sense; 3D radar sistemleri ve ADS-B transponder kaynaklarından gelen hava hedeflerine ait ham telemetri akışlarını gerçek zamanlı işleyen, hedefin icra ettiği 6 taktiksel uçuş manevrasını (*Düz Seyir, Koordineli Dönüş, Tırmanış, Dalış, Holding/Bekleme, Agresif Kaçış*) derin öğrenme ile anlık sınıflandıran ultra düşük gecikmeli ($\le 2\text{ ms}$) bir Taktiksel Durumsal Farkındalık Alt Sistemidir.

Sistem, hava hedeflerinden 8 boyutlu kinematik türevler çıkarır ve sıfır-tahsisli C++17 çıkarım çekirdeği üzerinde çalışan hibrit 1D-CNN + BiLSTM modeliyle 6 taktiksel manevrayı anlık olarak sınıflandırır.

---

## Sistem Mimarisi ve Temel Bileşenler

Sistem mimarisi dört ana işlem modülünden oluşmaktadır:

1. **Kinematik ve Telemetri İşleme Modülü (Streamer & Feature Extractor):**
   - Kesintili telemetri yayınlarında Kinematik Dead-Reckoning (ölü hesaplama) ile eksik zaman adımlarını ekstrapole eder.
   - $360^\circ \to 0^\circ$ açısal süreksizlik düzeltmeli 30 saniyelik hareketli dairesel kayan pencere yönetimi sağlar.
   - 8 boyutlu kinematik vektör hesaplar: Kartezyen koordinatlar ($x, y, z$), yer hızı ($v_g$), rota açısı ($\theta$), dönüş açısı hızı ($\omega$), teğetsel ivme ($a_t$) ve dikey ivme ($a_z$).

2. **AeroCore (C++17 Sıfır Bellek Tahsisli Çıkarım Çekirdeği):**
   - Gerçek zamanlı çalışma hattında `malloc`/`new` çağrılarını engelleyen statik boyutlu dairesel tampon (`RingBuffer<30, 8>`).
   - Her hedef izi için (`Track ID`) bağımsız kinematik türev ve durum yönetimi (`std::unordered_map<std::string, TrackContext>`).
   - AVX2/SIMD vektör hızlandırmalı ONNX çıkarım motoru ile milisaniye altı ($\sim 0.004\text{ ms}$) deterministik gecikme.

3. **Ağ Katmanı ve Protokol Serileştirme:**
   - Diller arası sıfır kayıplı telemetri aktarımı için Google Protocol Buffers v3 ikili paketleme.
   - Windows (`Winsock2`) ve Linux (`POSIX BSD Sockets`) ortamlarını şeffaf yöneten çapraz platform ham UDP soket sarmalayıcısı.
   - Çıkarım sonuçlarını 5006 portuna yüksek frekansta basan asenkron yayıncı.

4. **Taktik Komuta Kontrol ve Gözlemlenebilirlik (Docker InfluxDB & Grafana):**
   - Gelen telemetri ve manevra kararlarını InfluxDB 2.7 veritabanına Line Protocol ile yazan köprü servisi.
   - Hava sahasındaki hedefleri ve icra edilen manevraya göre renklenen rota izlerini anlık çizen Grafana GeoMap taktik haritası.
   - Çıkarım gecikmesi (ms), model tahmin güveni (%), irtifa profili ve hız zaman serisi göstergeleri.

---

## Teknolojik Altyapı

- **Yapay Zeka ve Modelleme:** PyTorch 2.x, ONNX Runtime, ONNX (Opset 17)
- **Çekirdek Çıkarım Motoru:** C++17, CMake 3.16+, MSVC (Windows) / GCC (Linux), AVX2 SIMD Optimizasyonu
- **Haberleşme ve Protokol:** Google Protocol Buffers v3, Ham UDP Soketleri (Winsock2 / POSIX)
- **Zaman Serisi Veritabanı ve Panel:** InfluxDB 2.7, Grafana OSS, Docker Compose
- **Sürekli Entegrasyon (CI/CD):** GitHub Actions (`ubuntu-22.04` ve `windows-latest` ikili test matrisi)

---

## Kinematik Durum Modeli (8 Boyutlu Vektör)

Çıkarım hattı, sürekli uçuş yörüngelerini 1 Hz frekansında örnekler ve 30 saniyelik hareketli bir zaman penceresi ($30 \times 8$ matris) tutar:

| İndeks | Sembol | Parametre Adı | Birim | Matematiksel Tanım |
|:---:|:---:|---|:---:|---|
| 0 | $x$ | Doğu Bağıl Konumu | m | Radara/referans noktasına göre kartezyen Doğu konumu |
| 1 | $y$ | Kuzey Bağıl Konumu | m | Radara/referans noktasına göre kartezyen Kuzey konumu |
| 2 | $z$ | İrtifa | m | Barometrik veya geometrik irtifa |
| 3 | $v_g$ | Yer Hızı | m/s | $v_g = \sqrt{\dot{x}^2 + \dot{y}^2}$ |
| 4 | $\theta$ | Rota / Yönelim Açısı | deg | $\theta = \text{atan2}(\dot{y}, \dot{x}) \pmod{360}$ |
| 5 | $\omega$ | Dönüş Açısı Hızı | deg/s | $\omega_t = (\theta_t - \theta_{t-1}) / \Delta t$ (Normalizasyon: $[-180^\circ, +180^\circ]$) |
| 6 | $a_t$ | Teğetsel İvme | $\text{m/s}^2$ | $a_{t} = (v_{g,t} - v_{g,t-1}) / \Delta t$ |
| 7 | $a_z$ | Dikey İvme | $\text{m/s}^2$ | $a_{z,t} = (v_{z,t} - v_{z,t-1}) / \Delta t$ |

---

## Taktik Manevra Taksonomisi

Sınıflandırıcı 6 operasyonel manevra sınıfını ayırt eder:

| ID | Sınıf Adı | Kinematik Belirteçler | Operasyonel Anlam |
|:---:|---|---|---|
| 0 | Düz Seyir (Straight Cruise) | $\omega \approx 0^\circ/\text{s}$, $a_t \approx 0$, $a_z \approx 0$, sabit hız/irtifa | Rutin intikal / devriye uçuşu |
| 1 | Koordineli Dönüş (Coordinated Turn) | $\|\omega\| \in [1.5^\circ/\text{s}, 4.0^\circ/\text{s}]$, $v_z \approx 0$ | Standart rota değişikliği |
| 2 | Tırmanış (Climb) | $v_z > +5.0\,\text{m/s}$, $a_z > 0$, $\omega \approx 0$ | İrtifa kazanma |
| 3 | Dalış / Alçalma (Descent / Dive) | $v_z < -5.0\,\text{m/s}$, $a_z < 0$, $\omega \approx 0$ | Taktiksel alçalma / iniş yaklaşması |
| 4 | Bekleme Turu (Orbit / Holding) | Sürekli sabit $\omega$, kapalı $360^\circ$ dairesel rota | Hava sahası bekleme / loitering |
| 5 | Agresif Kaçış (Evasive Maneuver) | $\|\omega\| > 6.0^\circ/\text{s}$, $\|a_t\| > 3.0\,\text{m/s}^2$, yüksek G-yükü | Tehdit kaçınma / jinking |

---

## Proje Dizin Yapısı

```
Aero-Sense/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Çoklu platform GitHub Actions CI hattı
├── core/                            # C++17 Çekirdek Çıkarım Motoru Paketi
│   ├── CMakeLists.txt               # Çapraz platform CMake derleme dosyası
│   ├── include/
│   │   ├── ring_buffer.hpp          # Sıfır-tahsis dairesel tampon başlığı
│   │   ├── onnx_engine.hpp          # Yüksek başarımlı çıkarım motoru sarmalayıcısı
│   │   └── udp_socket.hpp           # Çapraz platform ham UDP soket modülü
│   └── src/
│       └── main.cpp                 # Çoklu hedef canlı çıkarım döngüsü
├── data/
│   ├── raw/                         # Ham ADS-B ve radar kayıtları
│   └── processed/                   # İşlenmiş .npy veri setleri ve norm_params.json
├── docker/
│   ├── docker-compose.yml           # InfluxDB 2.7, Grafana ve Telemetri Köprü servisi
│   ├── Dockerfile.forwarder         # Köprü servis konteyner tanımı
│   ├── telemetry_forwarder.py       # UDP 5006 -> InfluxDB Line Protocol servisi
│   └── grafana/provisioning/        # Otomatik veri kaynağı ve dashboard ayarları
├── docs/
│   └── system_architecture.md       # Detaylı Sistem Mimarisi ve Kinematik Tasarım
├── models/
│   ├── model_best.pt                # Eğitilmiş PyTorch model kontrol noktası
│   └── model_cnn_lstm.onnx          # Production ONNX modeli
├── proto/
│   ├── telemetry.proto              # Google Protocol Buffers v3 şeması
│   └── generated/                   # Üretilen Python Protobuf bağlayıcıları
├── scripts/
│   ├── generate_synthetic_data.py   # 6 sınıflı 6DOF kinematik yörünge simülatörü
│   ├── train_model.py               # Derin öğrenme eğitimi ve ONNX export betiği
│   ├── opensky_streamer.py          # Kinematik dead-reckoning destekli ADS-B akıtıcı
│   └── test_end_to_end.py           # Otomatik uçtan uca entegrasyon ve gecikme testi
├── README.md                        # İki dilli proje dokümantasyonu
└── requirements.txt                 # Python paket bağımlılıkları
```

---

## Kurulum ve Kullanım

### 1. Bağımlılıkların Yüklenmesi

Gerekli Python paketlerini yüklemek için terminalde aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

### 2. Protobuf Şemasının Derlenmesi

Google Protocol Buffers şemasını derleyerek Python bağlayıcılarını üretin:

```bash
python -m grpc_tools.protoc -Iproto --python_out=proto/generated proto/telemetry.proto
```

### 3. Sentetik Veri Üretimi ve Model Eğitimi

6-DOF kinematik yörünge veri setini üretin ve 1D-CNN + BiLSTM modelini eğitin:

```bash
# 7.200 sentetik 6-DOF kinematik yörünge penceresi üretimi:
python scripts/generate_synthetic_data.py --samples_per_class 1200

# PyTorch eğitimi ve optimize ONNX modelinin dışa aktarılması:
python scripts/train_model.py --epochs 25
```

### 4. C++ Çekirdek Çıkarım Motorunun Derlenmesi

#### Windows (Visual Studio / MSVC):
```powershell
cd core
cmake -B build
cmake --build build --config Release
```

#### Linux (Ubuntu 22.04 LTS):
```bash
cd core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 5. Gözlemlenebilirlik Servislerinin Başlatılması (Docker Compose)

InfluxDB 2.7, Grafana ve telemetri köprü servisini başlatın:

```bash
cd docker
docker compose up -d
```

- **Grafana Taktik Paneli:** `http://localhost:3000` (Kullanıcı: `admin`, Şifre: `admin`)
- **InfluxDB Paneli:** `http://localhost:8086`

### 6. Canlı Çıkarım Hattının Çalıştırılması

İki ayrı terminal penceresi açın:

**1. Terminal: C++ Çekirdek Çıkarım Motoru**
```powershell
# Windows:
.\core\build\Release\maneuver_inference_engine.exe 5005 5006

# Linux:
./core/build/maneuver_inference_engine 5005 5006
```

**2. Terminal: Telemetri Akıtıcı (OpenSky ADS-B / Sentetik)**
```powershell
python scripts/opensky_streamer.py --host 127.0.0.1 --port 5005
```

### 7. Otomatik Uçtan Uca Benchmark Testi

Sistemin çıkarım gecikmesini ve manevra doğruluk skorlarını test edin:

```bash
python scripts/test_end_to_end.py
```

---

## Performans Kıyaslamaları ve Doğrulama

Otomatik entegrasyon ve gecikme testi sonuçları (`scripts/test_end_to_end.py`):

| Test Senaryosu | Gerçek Sınıf | Tahmin Edilen Sınıf | Model Güveni | Çekirdek Gecikmesi | Test Durumu |
|---|---|---|:---:|:---:|:---:|
| `TRK-CRUISE-01` | Düz Seyir | `STRAIGHT_CRUISE` | %91.6 | 0.004 ms | GEÇTİ |
| `TRK-TURN-02` | Koordineli Dönüş | `COORDINATED_TURN` | %99.3 | 0.003 ms | GEÇTİ |
| `TRK-CLIMB-03` | Tırmanış | `CLIMB` | %100.0 | 0.006 ms | GEÇTİ |
| `TRK-EVASIVE-04` | Agresif Kaçış | `EVASIVE_MANEUVER` | %100.0 | 0.002 ms | GEÇTİ |

- **Ortalama Çekirdek Gecikmesi:** 0.004 ms (4 mikrosaniye)
- **Maksimum Çekirdek Gecikmesi:** 0.006 ms
- **Savunma SLA Gereksinimi:** $\le 2.0\text{ ms}$ ($> 300\times$ tolerans marjıyla sağlandı)
- **Sayısal Eşlik (PyTorch vs. ONNX Runtime):** $MSE = 3.45 \times 10^{-13}$
