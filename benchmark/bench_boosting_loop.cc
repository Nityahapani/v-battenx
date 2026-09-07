#include "vbatten_x/learner.h"
#include "vbatten_x/data.h"
#include "vbatten_x/parameter.h"
#include "src/common/timer.h"
#include "src/vbatten_x_impl.cc"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using namespace vbx;

std::shared_ptr<PhysicalDataset> MakeDenseDataset(
    std::vector<vbx_float>, vbx_index, vbx_index,
    FeatureMap, PhysicalMetaInfo, std::vector<vbx_float>);

std::unique_ptr<VBattenLearner> MakeLearner(VBXParameter);

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    vbx_index nrows = 10000;
    vbx_index ncols = 20;
    std::vector<vbx_float> X(nrows * ncols), y(nrows);
    for (auto& v : X) v = nd(rng);
    for (auto& v : y) v = nd(rng);

    FeatureMap fm;
    for (int c = 0; c < ncols; ++c) fm.Add("f" + std::to_string(c));
    auto ds = MakeDenseDataset(X, nrows, ncols, std::move(fm), {}, y);

    struct Config { std::string name; std::string dtdo; int stages; };
    std::vector<Config> configs = {
        {"plain   (none)",       "none",      100},
        {"threshold DTDO",       "threshold", 100},
        {"router DTDO",          "router",    100},
        {"learned DTDO",         "learned",    50},
    };

    std::cout << std::left << std::setw(24) << "Config"
              << std::setw(10) << "Stages"
              << std::setw(14) << "Time(ms)"
              << std::setw(12) << "ms/stage"
              << "TrainLoss\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto& cfg : configs) {
        VBXParameter p;
        p.Set("dtdo",          cfg.dtdo);
        p.Set("learning_rate", 0.1);
        p.Set("tau_expand",    0.001);
        p.Set("verbose",       0);

        auto learner = MakeLearner(std::move(p));

        auto t0 = std::chrono::high_resolution_clock::now();
        learner->Train(*ds, cfg.stages, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::left << std::setw(24) << cfg.name
                  << std::setw(10) << learner->NumStages()
                  << std::setw(14) << std::fixed << std::setprecision(2) << ms
                  << std::setw(12) << std::setprecision(3) << ms / learner->NumStages()
                  << std::setprecision(6) << learner->TrainLoss() << "\n";
    }
    return 0;
}
