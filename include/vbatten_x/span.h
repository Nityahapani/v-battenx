#pragma once

#include <cstddef>
#include <stdexcept>

namespace vbx {

template <typename T>
class Span {
public:
    Span() : data_(nullptr), size_(0) {}
    Span(T* data, std::size_t size) : data_(data), size_(size) {}

    T* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T* begin() { return data_; }
    T* end()   { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end()   const { return data_ + size_; }

private:
    T* data_;
    std::size_t size_;
};

} // namespace vbx
