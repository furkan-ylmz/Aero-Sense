#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <vector>

namespace defense {
namespace telemetry {

/**
 * @brief Fixed-size, thread-safe Circular Ring Buffer for 30x8 Kinematic Telemetry Windows.
 * Adheres strictly to zero dynamic memory allocation on the hot path (deterministic latency).
 *
 * @tparam WINDOW_SIZE Number of sequential time steps (default: 30 seconds)
 * @tparam NUM_FEATURES Number of kinematic features per time step (default: 8)
 */
template <size_t WINDOW_SIZE = 30, size_t NUM_FEATURES = 8>
class RingBuffer {
public:
    RingBuffer() : head_(0), count_(0) {
        buffer_.fill(0.0f);
    }

    /**
     * @brief Pushes a new 8-feature sample into the ring buffer.
     * Overwrites the oldest sample once capacity (30) is reached.
     */
    void push(const std::array<float, NUM_FEATURES>& sample) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t f = 0; f < NUM_FEATURES; ++f) {
            buffer_[head_ * NUM_FEATURES + f] = sample[f];
        }
        head_ = (head_ + 1) % WINDOW_SIZE;
        if (count_ < WINDOW_SIZE) {
            count_++;
        }
    }

    /**
     * @brief Checks if the buffer has accumulated at least WINDOW_SIZE samples.
     */
    bool is_ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ >= WINDOW_SIZE;
    }

    /**
     * @brief Current number of samples stored in buffer (max WINDOW_SIZE).
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    /**
     * @brief Copies the chronological window (oldest to newest) into a flat pre-allocated buffer.
     * @param out_buffer Pointer to pre-allocated float buffer of size (WINDOW_SIZE * NUM_FEATURES)
     */
    void get_ordered_window(float* out_buffer) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ < WINDOW_SIZE) {
            // Buffer not full yet, pad older entries with zeros
            size_t missing = WINDOW_SIZE - count_;
            std::fill(out_buffer, out_buffer + (missing * NUM_FEATURES), 0.0f);
            for (size_t i = 0; i < count_; ++i) {
                size_t src_idx = (head_ + i) % WINDOW_SIZE;
                for (size_t f = 0; f < NUM_FEATURES; ++f) {
                    out_buffer[(missing + i) * NUM_FEATURES + f] = buffer_[src_idx * NUM_FEATURES + f];
                }
            }
        } else {
            // Full window: start reading from current 'head_' (which points to oldest entry)
            for (size_t i = 0; i < WINDOW_SIZE; ++i) {
                size_t src_idx = (head_ + i) % WINDOW_SIZE;
                for (size_t f = 0; f < NUM_FEATURES; ++f) {
                    out_buffer[i * NUM_FEATURES + f] = buffer_[src_idx * NUM_FEATURES + f];
                }
            }
        }
    }

    /**
     * @brief Resets buffer contents and counters.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.fill(0.0f);
        head_ = 0;
        count_ = 0;
    }

private:
    mutable std::mutex mutex_;
    std::array<float, WINDOW_SIZE * NUM_FEATURES> buffer_;
    size_t head_;
    size_t count_;
};

} // namespace telemetry
} // namespace defense
