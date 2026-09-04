#pragma once

#include <vector>
#include "base.h"

namespace vbx {

template <typename T>
class HostDeviceVector {
public:
    HostDeviceVector() = default;
    explicit HostDeviceVector(std::size_t n, T val = T{}) : host_(n, val) {}

    std::vector<T>&       HostVector()       { return host_; }
    const std::vector<T>& HostVector() const { return host_; }

    T*          Data()       { return host_.data(); }
    const T*    Data() const { return host_.data(); }
    std::size_t Size() const { return host_.size(); }
    bool        Empty() const { return host_.empty(); }

    void Resize(std::size_t n, T val = T{}) { host_.resize(n, val); }
    void PushBack(const T& v) { host_.push_back(v); }

    T& operator[](std::size_t i)       { return host_[i]; }
    const T& operator[](std::size_t i) const { return host_[i]; }

    typename std::vector<T>::iterator begin() { return host_.begin(); }
    typename std::vector<T>::iterator end()   { return host_.end(); }

private:
    std::vector<T> host_;
};

} // namespace vbx
