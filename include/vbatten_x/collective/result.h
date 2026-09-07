#pragma once
#include <string>

namespace vbx {

struct CollectiveResult {
    bool        success = true;
    std::string error_message;

    static CollectiveResult Ok()               { return {true,  ""}; }
    static CollectiveResult Err(std::string m) { return {false, std::move(m)}; }

    explicit operator bool() const { return success; }
};

} // namespace vbx
