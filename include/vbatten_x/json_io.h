#pragma once

#include "json.h"
#include "parameter.h"
#include <string>
#include <fstream>
#include <stdexcept>

namespace vbx {

inline JsonValue ParameterToJson(const VBXParameter& p) {
    auto obj = JsonValue::Object();
    for (auto& [k, v] : p.All()) {
        std::visit([&](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, int>)         obj.Set(k, JsonValue(val));
            else if constexpr (std::is_same_v<T, double>)  obj.Set(k, JsonValue(val));
            else if constexpr (std::is_same_v<T, bool>)    obj.Set(k, JsonValue(val));
            else if constexpr (std::is_same_v<T, std::string>) obj.Set(k, JsonValue(val));
        }, v);
    }
    return obj;
}

inline VBXParameter JsonToParameter(const JsonValue& j) {
    VBXParameter p;
    for (auto& [k, v] : j.Object()) {
        if (v.IsBool())        p.Set(k, v.AsBool());
        else if (v.IsInt())    p.Set(k, v.AsInt());
        else if (v.IsDouble()) p.Set(k, v.AsDouble());
        else if (v.IsString()) p.Set(k, v.AsString());
    }
    return p;
}

inline std::string ReadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

inline void WriteFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write file: " + path);
    f << content;
}

} // namespace vbx
