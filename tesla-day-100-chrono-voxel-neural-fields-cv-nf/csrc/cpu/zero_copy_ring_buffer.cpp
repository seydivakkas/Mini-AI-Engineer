/**
 * Tesla Chrono-Voxel Neural Fields (CV-NF)
 * Zero-Copy IPC Lock-Free Ring Buffer for HW4 Neuromorphic & RGB Pipeline
 *
 * Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
 * All Rights Reserved.
 */

#include <atomic>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

struct EventPacket {
    int32_t x;
    int32_t y;
    float timestamp_s;
    float polarity;
};

class TeslaZeroCopyEventRingBuffer {
private:
    std::vector<EventPacket> m_buffer;
    size_t m_capacity;
    std::atomic<size_t> m_head{0};
    std::atomic<size_t> m_tail{0};

public:
    explicit TeslaZeroCopyEventRingBuffer(size_t capacity = 1048576)
        : m_capacity(capacity), m_buffer(capacity) {}

    bool push_event(int32_t x, int32_t y, float t, float p) {
        size_t current_head = m_head.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) % m_capacity;

        if (next_head == m_tail.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }

        m_buffer[current_head] = {x, y, t, p};
        m_head.store(next_head, std::memory_order_release);
        return true;
    }

    size_t pop_batch(EventPacket* dest, size_t max_count) {
        size_t current_tail = m_tail.load(std::memory_order_relaxed);
        size_t current_head = m_head.load(std::memory_order_acquire);

        size_t count = 0;
        while (current_tail != current_head && count < max_count) {
            dest[count++] = m_buffer[current_tail];
            current_tail = (current_tail + 1) % m_capacity;
        }

        m_tail.store(current_tail, std::memory_order_release);
        return count;
    }

    size_t size() const {
        size_t h = m_head.load(std::memory_order_relaxed);
        size_t t = m_tail.load(std::memory_order_relaxed);
        if (h >= t) return h - t;
        return m_capacity - (t - h);
    }
};
