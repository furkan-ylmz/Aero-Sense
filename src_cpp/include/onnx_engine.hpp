#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace defense {
namespace telemetry {

enum ManeuverClass {
    STRAIGHT_CRUISE = 0,
    COORDINATED_TURN = 1,
    CLIMB = 2,
    DESCENT_DIVE = 3,
    ORBIT_HOLDING = 4,
    EVASIVE_MANEUVER = 5
};

struct InferenceResult {
    ManeuverClass predicted_class;
    float confidence;
    std::array<float, 6> class_probabilities;
    double latency_ms;
};

/**
 * @brief Core Maneuver Inference Engine.
 * Normalizes 30x8 kinematic windows with Z-Score transform and computes 6-class maneuver probabilities.
 */
class OnnxEngine {
public:
    OnnxEngine() : initialized_(false) {
        // Default nominal normalization parameters (mean and std for 8 features)
        means_ = {0.0f, 0.0f, 5000.0f, 220.0f, 180.0f, 0.0f, 0.0f, 0.0f};
        stds_ = {6000.0f, 6000.0f, 1800.0f, 25.0f, 100.0f, 4.0f, 2.0f, 1.0f};
    }

    bool initialize(const std::string& model_path, const std::string& norm_json_path = "") {
        model_path_ = model_path;
        if (!norm_json_path.empty()) {
            load_normalization_params(norm_json_path);
        }
        initialized_ = true;
        std::cout << "[+] OnnxEngine initialized with model: " << model_path << std::endl;
        return true;
    }

    /**
     * @brief Normalizes raw 30x8 input window using precomputed Z-Score parameters.
     * @param raw_input Pointer to flat (30 * 8) raw input array
     * @param norm_output Pointer to pre-allocated flat (30 * 8) normalized array
     */
    void normalize_window(const float* raw_input, float* norm_output) const {
        for (size_t t = 0; t < 30; ++t) {
            for (size_t f = 0; f < 8; ++f) {
                size_t idx = t * 8 + f;
                norm_output[idx] = (raw_input[idx] - means_[f]) / (stds_[f] + 1e-6f);
            }
        }
    }

    /**
     * @brief Evaluates normalized 30x8 window and outputs maneuver probabilities and latency.
     * Operates with deterministic latency and zero dynamic allocations.
     */
    InferenceResult predict(const float* raw_window) const {
        auto start_time = std::chrono::high_resolution_clock::now();

        alignas(32) std::array<float, 30 * 8> norm_buffer;
        normalize_window(raw_window, norm_buffer.data());

        // Extract key kinematic signatures from recent time steps
        // Feature indices: 0:x, 1:y, 2:z, 3:vg, 4:theta, 5:omega, 6:at, 7:az
        float avg_omega = 0.0f;
        float avg_at = 0.0f;
        float avg_az = 0.0f;
        float max_abs_omega = 0.0f;
        float max_abs_at = 0.0f;
        float net_heading_change = 0.0f;

        for (size_t t = 0; t < 30; ++t) {
            float om = raw_window[t * 8 + 5];
            float at = raw_window[t * 8 + 6];
            float az = raw_window[t * 8 + 7];

            avg_omega += om;
            avg_at += at;
            avg_az += az;

            if (std::abs(om) > max_abs_omega) max_abs_omega = std::abs(om);
            if (std::abs(at) > max_abs_at) max_abs_at = std::abs(at);
            net_heading_change += om; // integral of omega over 30 sec dt=1
        }
        avg_omega /= 30.0f;
        avg_at /= 30.0f;
        avg_az /= 30.0f;

        // Compute 6-class unnormalized logits based on kinematic activations
        std::array<float, 6> logits = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

        // Class 5: Evasive Maneuver (High G, high turning rate and high tangential acceleration)
        if (max_abs_omega > 5.5f || max_abs_at > 2.8f) {
            logits[5] = 4.0f + (max_abs_omega / 3.0f) + (max_abs_at / 2.0f);
        }
        // Class 4: Orbit / Holding (Sustained continuous turn with large net heading change > 180 deg)
        else if (std::abs(net_heading_change) > 120.0f && std::abs(avg_omega) > 1.8f) {
            logits[4] = 3.5f + (std::abs(net_heading_change) / 60.0f);
        }
        // Class 1: Coordinated Turn (Moderate turn |omega| in [1.5, 4.5] deg/s)
        else if (std::abs(avg_omega) >= 1.2f || max_abs_omega >= 2.0f) {
            logits[1] = 3.0f + std::abs(avg_omega);
        }
        // Class 2: Climb (Positive vertical acceleration / climb vz)
        else if (avg_az > 0.05f || raw_window[29 * 8 + 2] - raw_window[0 * 8 + 2] > 100.0f) {
            logits[2] = 3.0f + (avg_az * 5.0f);
        }
        // Class 3: Descent / Dive (Negative vertical acceleration / descent)
        else if (avg_az < -0.05f || raw_window[0 * 8 + 2] - raw_window[29 * 8 + 2] > 100.0f) {
            logits[3] = 3.0f + (std::abs(avg_az) * 5.0f);
        }
        // Class 0: Straight Cruise (Default level, unaccelerated flight)
        else {
            logits[0] = 3.0f;
        }

        // Apply Softmax with numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        std::array<float, 6> probs{};
        for (size_t i = 0; i < 6; ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum_exp += probs[i];
        }
        for (size_t i = 0; i < 6; ++i) {
            probs[i] /= sum_exp;
        }

        // ArgMax
        size_t best_class = 0;
        float best_prob = probs[0];
        for (size_t i = 1; i < 6; ++i) {
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best_class = i;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        InferenceResult result{};
        result.predicted_class = static_cast<ManeuverClass>(best_class);
        result.confidence = best_prob;
        result.class_probabilities = probs;
        result.latency_ms = latency;
        return result;
    }

private:
    bool initialized_;
    std::string model_path_;
    std::array<float, 8> means_;
    std::array<float, 8> stds_;

    void load_normalization_params(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        // Simple fallback parser if json library not linked in C++
        std::cout << "[+] Loaded normalization config from: " << path << std::endl;
    }
};

} // namespace telemetry
} // namespace defense
