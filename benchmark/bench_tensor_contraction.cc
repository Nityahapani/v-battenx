#include "src/field/tensor/contraction_engine.cc"
#include "src/field/tensor/rank_adaptive_tensor.cc"
#include "src/common/timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

using namespace vbx;

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<int> dims = {4, 8, 16, 32, 64};
    int reps              = 1000;

    std::cout << std::left << std::setw(10) << "Dim"
              << std::setw(14) << "Time(ms)"
              << "us/call\n";
    std::cout << std::string(34, '-') << "\n";

    for (int dim : dims) {
        auto T = MakeRank2Tensor(0, dim, dim);
        {
            auto d = T->Data();
            for (auto& v : d) v = ud(rng);
        }

        // ContractWith needs a rank-1 "other" — use a 1-column tensor
        auto other = MakeRank2Tensor(0, dim, 1);
        {
            auto d = other->Data();
            for (auto& v : d) v = ud(rng);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i)
            T->ContractWith(*other);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::left << std::setw(10) << dim
                  << std::setw(14) << std::fixed << std::setprecision(3) << ms
                  << std::setprecision(3) << (ms / reps * 1000.0) << "\n";
    }
    return 0;
}
