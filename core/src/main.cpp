#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ring_buffer.hpp"
#include "onnx_engine.hpp"
#include "udp_socket.hpp"

using namespace defense::telemetry;

const char* MANEUVER_STRINGS[] = {
    "STRAIGHT_CRUISE",
    "COORDINATED_TURN",
    "CLIMB",
    "DESCENT_DIVE",
    "ORBIT_HOLDING",
    "EVASIVE_MANEUVER"
};

// Binary Telemetry Packet format for UDP transport (matching telemetry_forwarder.py)
#pragma pack(push, 1)
struct RawInputPacket {
    char track_id[16];
    uint64_t timestamp_ms;
    double lat;
    double lon;
    double alt_m;
    double speed_mps;
    double heading_deg;
};

struct OutputClassificationPacket {
    char track_id[16];
    uint64_t timestamp_ms;
    double lat;
    double lon;
    double alt_m;
    double speed_mps;
    double heading_deg;
    int32_t maneuver_id;
    float confidence;
    float latency_ms;
};
#pragma pack(pop)

struct TrackContext {
    RingBuffer<30, 8> ring_buffer;
    double prev_lat = 0.0;
    double prev_lon = 0.0;
    double prev_alt = 0.0;
    double prev_speed = 0.0;
    double prev_heading = 0.0;
    bool has_prev = false;
};

int main(int argc, char* argv[]) {
    uint16_t input_port = 5005;
    uint16_t output_port = 5006;
    std::string forwarder_host = "127.0.0.1";

    if (argc > 1) input_port = static_cast<uint16_t>(std::atoi(argv[1]));
    if (argc > 2) output_port = static_cast<uint16_t>(std::atoi(argv[2]));

    std::cout << "=========================================================" << std::endl;
    std::cout << "  AERO-SENSE: REAL-TIME TACTICAL INFERENCE ENGINE (C++17)" << std::endl;
    std::cout << "  Platform: Cross-Platform (Windows Native & Linux)" << std::endl;
    std::cout << "  Target Latency Budget: <= 2.0 ms" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // Initialize modules
    OnnxEngine engine;
    engine.initialize("models/model_cnn_lstm.onnx", "data/processed/norm_params.json");

    UdpSocket in_socket;
    if (!in_socket.bind_port(input_port)) {
        std::cerr << "[!] Failed to bind UDP receiver on port " << input_port << std::endl;
        return 1;
    }
    in_socket.set_receive_timeout(500); // 500ms timeout for graceful loop

    UdpSocket out_socket;
    std::cout << "[+] Listening for Telemetry on UDP port " << input_port << "..." << std::endl;
    std::cout << "[+] Broadcasting Maneuver Predictions to " << forwarder_host << ":" << output_port << std::endl;

    alignas(32) std::array<float, 30 * 8> window_buffer{};
    std::unordered_map<std::string, TrackContext> active_tracks;
    uint64_t processed_frames = 0;

    char recv_buf[1024];

    while (true) {
        std::string sender_ip;
        int bytes = in_socket.receive_from(recv_buf, sizeof(recv_buf), &sender_ip);

        if (bytes <= 0) {
            // Timeout: wait for next packet
            continue;
        }

        auto pipeline_start = std::chrono::high_resolution_clock::now();

        // Parse Input Packet
        RawInputPacket in_pkt{};
        if (bytes >= static_cast<int>(sizeof(RawInputPacket))) {
            std::memcpy(&in_pkt, recv_buf, sizeof(RawInputPacket));
        } else {
            // Simulated default payload if raw payload length differs
            std::strncpy(in_pkt.track_id, "TRK-01", sizeof(in_pkt.track_id));
            in_pkt.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            in_pkt.lat = 39.9334;
            in_pkt.lon = 32.8597;
            in_pkt.alt_m = 7500.0;
            in_pkt.speed_mps = 230.0;
            in_pkt.heading_deg = 45.0;
        }

        std::string track_key(in_pkt.track_id, strnlen(in_pkt.track_id, 16));
        TrackContext& ctx = active_tracks[track_key];

        // Compute Kinematic Derivatives
        float dt = 1.0f;
        float d_heading = 0.0f;
        float omega = 0.0f;
        float at = 0.0f;
        float az = 0.0f;

        if (ctx.has_prev) {
            float raw_d_heading = static_cast<float>(in_pkt.heading_deg - ctx.prev_heading);
            // Normalize angle diff [-180, 180]
            d_heading = std::fmod(raw_d_heading + 180.0f, 360.0f) - 180.0f;
            if (d_heading < -180.0f) d_heading += 360.0f;
            omega = d_heading / dt;
            at = static_cast<float>(in_pkt.speed_mps - ctx.prev_speed) / dt;
            az = static_cast<float>(in_pkt.alt_m - ctx.prev_alt) / dt;
        }

        ctx.prev_lat = in_pkt.lat;
        ctx.prev_lon = in_pkt.lon;
        ctx.prev_alt = in_pkt.alt_m;
        ctx.prev_speed = in_pkt.speed_mps;
        ctx.prev_heading = in_pkt.heading_deg;
        ctx.has_prev = true;

        // Approx ENU meters from (lat, lon)
        float x_m = static_cast<float>(in_pkt.lon * 111000.0);
        float y_m = static_cast<float>(in_pkt.lat * 111000.0);
        float z_m = static_cast<float>(in_pkt.alt_m);
        float vg = static_cast<float>(in_pkt.speed_mps);
        float th = static_cast<float>(in_pkt.heading_deg);

        std::array<float, 8> sample = {x_m, y_m, z_m, vg, th, omega, at, az};
        ctx.ring_buffer.push(sample);
        ctx.ring_buffer.get_ordered_window(window_buffer.data());

        // Execute Zero-Allocation Inference
        InferenceResult result = engine.predict(window_buffer.data());

        auto pipeline_end = std::chrono::high_resolution_clock::now();
        double total_latency_ms = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start).count();

        // Broadcast Output Binary Telemetry Packet
        OutputClassificationPacket out_pkt{};
        std::memcpy(out_pkt.track_id, in_pkt.track_id, sizeof(out_pkt.track_id));
        out_pkt.timestamp_ms = in_pkt.timestamp_ms;
        out_pkt.lat = in_pkt.lat;
        out_pkt.lon = in_pkt.lon;
        out_pkt.alt_m = in_pkt.alt_m;
        out_pkt.speed_mps = in_pkt.speed_mps;
        out_pkt.heading_deg = in_pkt.heading_deg;
        out_pkt.maneuver_id = static_cast<int32_t>(result.predicted_class);
        out_pkt.confidence = result.confidence;
        out_pkt.latency_ms = static_cast<float>(total_latency_ms);

        out_socket.send_to(forwarder_host, output_port, &out_pkt, sizeof(out_pkt));
        processed_frames++;

        if (processed_frames % 5 == 0 || result.predicted_class == EVASIVE_MANEUVER) {
            std::cout << "[TRACK " << in_pkt.track_id << "] -> "
                      << std::left << std::setw(18) << MANEUVER_STRINGS[result.predicted_class]
                      << " | Conf: " << std::fixed << std::setprecision(2) << (result.confidence * 100.0f) << "%"
                      << " | Alt: " << std::setw(6) << static_cast<int>(in_pkt.alt_m) << "m"
                      << " | Latency: " << std::setprecision(3) << total_latency_ms << " ms"
                      << (total_latency_ms <= 2.0 ? " [OK <= 2ms]" : " [WARN]")
                      << std::endl;
        }
    }

    return 0;
}
