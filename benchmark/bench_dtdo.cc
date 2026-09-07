#include "vbatten_x/topological_operator.h"
#include "src/dtdo/operator_registry.cc"
#include "src/dtdo/complexity_cost.cc"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include "src/physics/residual_info.cc"
#include "src/common/timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace vbx;

std::unique_ptr<LatentField>   MakeContinuousField(std::size_t, std::size_t);
std::unique_ptr<FieldTopology> MakeRegionGraph();
std::unique_ptr<DimensionMap>  MakeUniformDim(vbx_dim_t, std::size_t, float);
std::unique_ptr<TensorField>   MakeRank2Tensor(vbx_region_id, vbx_dim_t, vbx_dim_t);

static FieldState MakeState(int dim) {
    FieldState s;
    s.F = MakeContinuousField(dim, 1);
    s.K = MakeRegionGraph();
    s.d = MakeUniformDim(dim, 1, 256.0f);
    s.T = MakeRank2Tensor(0, dim, dim);
    return s;
}

int main() {
    ResidualInfo residuals;
    residuals.pde_residual        = {0.2f};
    residuals.prediction_residual = {0.1f};

    std::vector<std::string> ops = {"threshold", "gradient", "router", "learned"};
    std::vector<int> dims        = {4, 8, 16, 32};
    int iterations               = 200;

    std::cout << std::left << std::setw(18) << "Operator"
              << std::setw(8)  << "Dim"
              << std::setw(14) << "Time(ms)"
              << "us/call\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto& op_name : ops) {
        for (int dim : dims) {
            auto op = MakeOperator(op_name, 0.1f, 0.01f);
            if (!op) continue;

            auto state  = MakeState(dim);
            auto budget = ComplexityCost::FromState(state, 64, 16, 32);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i)
                op->Apply(state, residuals, budget);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << std::left << std::setw(18) << op_name
                      << std::setw(8)  << dim
                      << std::setw(14) << std::fixed << std::setprecision(2) << ms
                      << std::setprecision(1) << (ms / iterations * 1000.0) << "\n";
        }
    }
    return 0;
}
