#include "vbatten_x/json.h"
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace vbx {

static void SkipWs(const std::string& s, std::size_t& i) {
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
}

static JsonValue ParseValue(const std::string& s, std::size_t& i);

static std::string ParseString(const std::string& s, std::size_t& i) {
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            ++i;
            if (s[i] == 'n') out += '\n';
            else if (s[i] == 't') out += '\t';
            else out += s[i];
        } else {
            out += s[i];
        }
        ++i;
    }
    ++i;
    return out;
}

static JsonValue ParseObject(const std::string& s, std::size_t& i) {
    auto obj = JsonValue::Object();
    ++i;
    SkipWs(s, i);
    if (s[i] == '}') { ++i; return obj; }
    while (true) {
        SkipWs(s, i);
        auto key = ParseString(s, i);
        SkipWs(s, i);
        ++i; // ':'
        SkipWs(s, i);
        obj.Set(key, ParseValue(s, i));
        SkipWs(s, i);
        if (s[i] == '}') { ++i; break; }
        ++i; // ','
    }
    return obj;
}

static JsonValue ParseArray(const std::string& s, std::size_t& i) {
    auto arr = JsonValue::Array();
    ++i;
    SkipWs(s, i);
    if (s[i] == ']') { ++i; return arr; }
    while (true) {
        SkipWs(s, i);
        arr.Append(ParseValue(s, i));
        SkipWs(s, i);
        if (s[i] == ']') { ++i; break; }
        ++i; // ','
    }
    return arr;
}

static JsonValue ParseValue(const std::string& s, std::size_t& i) {
    SkipWs(s, i);
    char c = s[i];
    if (c == '"') return JsonValue(ParseString(s, i));
    if (c == '{') return ParseObject(s, i);
    if (c == '[') return ParseArray(s, i);
    if (s.substr(i, 4) == "true")  { i += 4; return JsonValue(true); }
    if (s.substr(i, 5) == "false") { i += 5; return JsonValue(false); }
    if (s.substr(i, 4) == "null")  { i += 4; return JsonValue(); }
    // number
    std::size_t start = i;
    bool is_double = false;
    if (s[i] == '-') ++i;
    while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
        if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') is_double = true;
        ++i;
    }
    std::string num = s.substr(start, i - start);
    if (is_double) return JsonValue(std::stod(num));
    return JsonValue(std::stoi(num));
}

JsonValue JsonParse(const std::string& s) {
    std::size_t i = 0;
    return ParseValue(s, i);
}

static void Indent(std::ostringstream& os, int n) {
    for (int i = 0; i < n; ++i) os << "  ";
}

std::string JsonValue::Dump(int indent) const {
    std::ostringstream os;
    switch (type_) {
        case Type::Null:   os << "null"; break;
        case Type::Bool:   os << (bool_ ? "true" : "false"); break;
        case Type::Int:    os << int_; break;
        case Type::Double: os << double_; break;
        case Type::String: {
            os << '"';
            for (char c : str_) {
                if (c == '"') os << "\\\"";
                else if (c == '\n') os << "\\n";
                else os << c;
            }
            os << '"';
            break;
        }
        case Type::Array: {
            os << "[\n";
            for (std::size_t k = 0; k < arr_.size(); ++k) {
                Indent(os, indent + 1);
                os << arr_[k].Dump(indent + 1);
                if (k + 1 < arr_.size()) os << ",";
                os << "\n";
            }
            Indent(os, indent);
            os << "]";
            break;
        }
        case Type::Object: {
            os << "{\n";
            std::size_t k = 0;
            for (auto& [key, val] : obj_) {
                Indent(os, indent + 1);
                os << '"' << key << "\": " << val.Dump(indent + 1);
                if (++k < obj_.size()) os << ",";
                os << "\n";
            }
            Indent(os, indent);
            os << "}";
            break;
        }
    }
    return os.str();
}

} // namespace vbx
