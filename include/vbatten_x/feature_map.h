#pragma once

#include "base.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace vbx {

enum class FeatureDtype { Float32, Float64, Int32, Int64 };

struct FeatureEntry {
    std::string  name;
    vbx_index    col_index;
    FeatureDtype dtype;
    std::string  unit;
    bool         is_positive = false;
};

class FeatureMap {
public:
    void Add(const std::string& name, FeatureDtype dtype = FeatureDtype::Float32,
             const std::string& unit = "", bool is_positive = false) {
        vbx_index idx = static_cast<vbx_index>(entries_.size());
        name_to_idx_[name] = idx;
        entries_.push_back({name, idx, dtype, unit, is_positive});
    }

    vbx_index IndexOf(const std::string& name) const {
        auto it = name_to_idx_.find(name);
        if (it == name_to_idx_.end()) throw std::runtime_error("Feature not found: " + name);
        return it->second;
    }

    const FeatureEntry& At(vbx_index i) const { return entries_.at(i); }
    std::size_t Size() const { return entries_.size(); }

    const std::vector<FeatureEntry>& Entries() const { return entries_; }

private:
    std::vector<FeatureEntry>              entries_;
    std::unordered_map<std::string, vbx_index> name_to_idx_;
};

} // namespace vbx
