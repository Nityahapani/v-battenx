#include "vbatten_x/learner.h"
#include "vbatten_x/data.h"
#include "vbatten_x/parameter.h"
#include "vbatten_x/json_io.h"
#include "vbatten_x/physics_spec.h"
#include "src/field/field_impls.h"
#include <cstring>
#include <string>
#include <memory>
#include <vector>

namespace vbx {
std::shared_ptr<PhysicalDataset> MakeDenseDataset(
    std::vector<vbx_float>, vbx_index, vbx_index,
    FeatureMap, PhysicalMetaInfo, std::vector<vbx_float>);
std::unique_ptr<VBattenLearner> MakeLearner(VBXParameter);

class VBattenLearnerImpl;
void SetPhysicsOnLearner(VBattenLearner* learner, const PhysicsSpec& spec);
}

using namespace vbx;

struct VBXHandle {
    std::unique_ptr<VBattenLearner>  learner;
    std::shared_ptr<PhysicalDataset> dataset;
};

static thread_local std::string g_last_error;

extern "C" {

void* vbx_learner_create(const char* params_json) {
    try {
        auto j = JsonParse(std::string(params_json));
        auto* h = new VBXHandle();
        h->learner = MakeLearner(JsonToParameter(j));
        return h;
    } catch (std::exception& e) { g_last_error = e.what(); return nullptr; }
}

int vbx_set_data(void* handle, const float* X, const float* y,
                 long nrows, long ncols) {
    try {
        auto* h = static_cast<VBXHandle*>(handle);
        std::vector<vbx_float> data(X, X + nrows * ncols);
        std::vector<vbx_float> labels(y, y + nrows);
        FeatureMap fm;
        for (long c = 0; c < ncols; ++c) fm.Add("f" + std::to_string(c));
        h->dataset = MakeDenseDataset(std::move(data), nrows, ncols,
                                      std::move(fm), {}, std::move(labels));
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

int vbx_set_physics(void* handle, const char* spec_json) {
    try {
        auto* h    = static_cast<VBXHandle*>(handle);
        auto  j    = JsonParse(std::string(spec_json));
        auto  spec = PhysicsSpec::FromJson(j);
        SetPhysicsOnLearner(h->learner.get(), spec);
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

int vbx_train(void* handle, int n_iters) {
    try {
        auto* h = static_cast<VBXHandle*>(handle);
        h->learner->Train(*h->dataset, n_iters, nullptr);
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

int vbx_predict(void* handle, const float* X, long nrows, long ncols,
                float* out) {
    try {
        auto* h = static_cast<VBXHandle*>(handle);
        std::vector<vbx_float> data(X, X + nrows * ncols);
        FeatureMap fm;
        for (long c = 0; c < ncols; ++c) fm.Add("f" + std::to_string(c));
        auto ds   = MakeDenseDataset(std::move(data), nrows, ncols,
                                     std::move(fm), {}, {});
        auto pred = h->learner->Predict(*ds);
        std::memcpy(out, pred.data(), pred.size() * sizeof(float));
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

int vbx_save(void* handle, const char* path) {
    try { static_cast<VBXHandle*>(handle)->learner->Save(path); return 0; }
    catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

int vbx_load(void* handle, const char* path) {
    try { static_cast<VBXHandle*>(handle)->learner->Load(path); return 0; }
    catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

float       vbx_train_loss(void* h) { return static_cast<VBXHandle*>(h)->learner->TrainLoss(); }
int         vbx_num_stages(void* h) { return static_cast<VBXHandle*>(h)->learner->NumStages(); }
void        vbx_destroy(void* h)    { delete static_cast<VBXHandle*>(h); }
const char* vbx_last_error()        { return g_last_error.c_str(); }

} // extern "C"
