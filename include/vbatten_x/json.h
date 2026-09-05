#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <stdexcept>
#include <sstream>
#include <memory>

namespace vbx {

class JsonValue {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    JsonValue() : type_(Type::Null) {}
    explicit JsonValue(bool v)               : type_(Type::Bool),   bool_(v) {}
    explicit JsonValue(int v)                : type_(Type::Int),    int_(v) {}
    explicit JsonValue(double v)             : type_(Type::Double), double_(v) {}
    explicit JsonValue(const std::string& v) : type_(Type::String), str_(v) {}
    explicit JsonValue(std::string&& v)      : type_(Type::String), str_(std::move(v)) {}

    static JsonValue Array()  { JsonValue j; j.type_ = Type::Array;  return j; }
    static JsonValue Object() { JsonValue j; j.type_ = Type::Object; return j; }

    Type GetType() const { return type_; }
    bool IsNull()   const { return type_ == Type::Null; }
    bool IsString() const { return type_ == Type::String; }
    bool IsInt()    const { return type_ == Type::Int; }
    bool IsDouble() const { return type_ == Type::Double || type_ == Type::Int; }
    bool IsBool()   const { return type_ == Type::Bool; }
    bool IsObject() const { return type_ == Type::Object; }
    bool IsArray()  const { return type_ == Type::Array; }

    bool        AsBool()   const { return bool_; }
    int         AsInt()    const { return int_; }
    double      AsDouble() const { return type_ == Type::Int ? (double)int_ : double_; }
    const std::string& AsString() const { return str_; }

    void Append(JsonValue v) { arr_.push_back(std::move(v)); }
    void Set(const std::string& key, JsonValue v) { obj_[key] = std::move(v); }

    bool Has(const std::string& key) const { return obj_.count(key) > 0; }

    const JsonValue& operator[](const std::string& key) const {
        auto it = obj_.find(key);
        if (it == obj_.end()) throw std::runtime_error("JSON key not found: " + key);
        return it->second;
    }
    const JsonValue& operator[](std::size_t i) const { return arr_[i]; }
    std::size_t ArraySize() const { return arr_.size(); }

    const std::unordered_map<std::string, JsonValue>& Object() const { return obj_; }

    std::string Dump(int indent = 0) const;

private:
    Type type_;
    bool bool_{};
    int  int_{};
    double double_{};
    std::string str_;
    std::vector<JsonValue> arr_;
    std::unordered_map<std::string, JsonValue> obj_;
};

JsonValue JsonParse(const std::string& s);

} // namespace vbx
