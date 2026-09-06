#include <cassert>
#include <cmath>
#include <cstdio>
#include "src/field/dimension/dim_transitions.h"
#include "src/field/dimension/dim_transitions.cc"

using namespace vbx;

static float FrobeniusNorm(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x * x;
    return static_cast<float>(std::sqrt(s));
}

static void TestExpand1dTo2dPreservesNorm() {
    std::vector<float> p = {1.0f, 2.0f, 3.0f, 4.0f};
    float norm_before = FrobeniusNorm(p);
    auto  expanded    = Expand1dTo2d(p, 42);
    float norm_after  = FrobeniusNorm(expanded);
    float rel_err     = std::abs(norm_after - norm_before) / (norm_before + 1e-9f);
    assert(rel_err < 0.01f && "Expand1dTo2d: Frobenius norm not preserved");
    assert(expanded.size() == 2 * p.size() && "Expand1dTo2d: size mismatch");
    printf("[PASS] Expand1dTo2d preserves norm (rel_err=%.6f)\n", rel_err);
}

static void TestExpand2dTo3dPreservesNorm() {
    std::vector<float> p = {1.0f, 2.0f, 3.0f, 4.0f};
    float norm_before = FrobeniusNorm(p);
    auto  expanded    = Expand2dTo3d(p, 42);
    float norm_after  = FrobeniusNorm(expanded);
    float rel_err     = std::abs(norm_after - norm_before) / (norm_before + 1e-9f);
    assert(rel_err < 0.01f && "Expand2dTo3d: Frobenius norm not preserved");
    printf("[PASS] Expand2dTo3d preserves norm (rel_err=%.6f)\n", rel_err);
}

static void TestCollapse3dTo2dRoundTrip() {
    std::vector<float> p = {1.0f, 0.5f, 0.0f,
                             0.0f, 1.0f, 0.0f,
                             0.3f, 0.2f, 0.9f,
                             0.1f, 0.4f, 0.6f};
    float norm_before = FrobeniusNorm(p);
    auto  collapsed   = Collapse3dTo2d(p);
    assert(collapsed.size() == 8 && "Collapse3dTo2d: size wrong");
    float norm_after = FrobeniusNorm(collapsed);
    float rel_err    = std::abs(norm_after - norm_before) / (norm_before + 1e-9f);
    assert(rel_err < 0.05f && "Collapse3dTo2d: too much energy lost");
    printf("[PASS] Collapse3dTo2d energy preserved (rel_err=%.6f)\n", rel_err);
}

static void TestCollapse2dTo1dRoundTrip() {
    std::vector<float> p = {1.0f, 0.5f,  0.3f, 0.8f,
                             0.2f, 0.9f,  0.7f, 0.1f};
    float norm_before = FrobeniusNorm(p);
    auto  collapsed   = Collapse2dTo1d(p);
    assert(collapsed.size() == 4 && "Collapse2dTo1d: size wrong");
    float norm_after = FrobeniusNorm(collapsed);
    float rel_err    = std::abs(norm_after - norm_before) / (norm_before + 1e-9f);
    assert(rel_err < 0.05f && "Collapse2dTo1d: too much energy lost");
    printf("[PASS] Collapse2dTo1d energy preserved (rel_err=%.6f)\n", rel_err);
}

static void TestLocalDimChangeNoOp() {
    std::vector<float> p = {1.0f, 2.0f, 3.0f, 4.0f};
    auto result = LocalDimChange(p, 2, 2, 42);
    assert(result == p && "LocalDimChange no-op should return same params");
    printf("[PASS] LocalDimChange no-op\n");
}

static void TestLocalDimChange1To3() {
    std::vector<float> p = {1.0f, 2.0f, 3.0f, 4.0f};
    float norm_before = FrobeniusNorm(p);
    auto  result      = LocalDimChange(p, 1, 3, 42);
    float norm_after  = FrobeniusNorm(result);
    float rel_err     = std::abs(norm_after - norm_before) / (norm_before + 1e-9f);
    assert(rel_err < 0.01f && "LocalDimChange 1->3: norm not preserved");
    printf("[PASS] LocalDimChange 1->3 norm preserved (rel_err=%.6f)\n", rel_err);
}

int main() {
    TestExpand1dTo2dPreservesNorm();
    TestExpand2dTo3dPreservesNorm();
    TestCollapse3dTo2dRoundTrip();
    TestCollapse2dTo1dRoundTrip();
    TestLocalDimChangeNoOp();
    TestLocalDimChange1To3();
    printf("\nAll dim_transition tests passed.\n");
    return 0;
}
