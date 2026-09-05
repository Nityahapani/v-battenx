#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#include <sstream>

namespace vbx {

using ParamValue = std::variant<int, double, std::string, bool>;

class VBXParameter {
public:
    void Set(const std::string& key, int v)                { store_[key] = v; }
    void Set(const std::string& key, double v)             { store_[key] = v; }
    void Set(const std::string& key, const std::string& v) { store_[key] = v; }
    void Set(const std::string& key, bool v)               { store_[key] = v; }

    template <typename T>
    T Get(const std::string& key) const {
        auto it = store_.find(key);
        if (it == store_.end()) throw std::runtime_error("Parameter not found: " + key);
        return std::get<T>(it->second);
    }

    template <typename T>
    T GetOr(const std::string& key, T def) const {
        auto it = store_.find(key);
        if (it == store_.end()) return def;
        return std::get<T>(it->second);
    }

    bool Has(const std::string& key) const { return store_.count(key) > 0; }

    const std::unordered_map<std::string, ParamValue>& All() const { return store_; }

private:
    std::unordered_map<std::string, ParamValue> store_;
};

} // namespace vbx
