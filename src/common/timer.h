#pragma once
#include <chrono>
#include <string>
#include <iostream>

namespace vbx {

struct ScopedTimer {
    using Clock = std::chrono::high_resolution_clock;
    std::string label;
    Clock::time_point t0;

    explicit ScopedTimer(std::string lbl)
        : label(std::move(lbl)), t0(Clock::now()) {}

    ~ScopedTimer() {
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
        std::cout << "[timer] " << label << ": " << ms << " ms\n";
    }

    double ElapsedMs() const {
        return std::chrono::duration<double, std::milli>(
                   Clock::now() - t0).count();
    }
};

} // namespace vbx
