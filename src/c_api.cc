#include "vbatten_x/learner.h"
#include "vbatten_x/data.h"
#include "vbatten_x/parameter.h"
#include "vbatten_x/json_io.h"
#include "vbatten_x/physics_spec.h"
#include "vbatten_x/version.h"
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
void SetPhysicsOnLearner(VBattenLearner*, const PhysicsSpec&);
}

using namespace vbx;

struct VBXHandle {
    std::unique_ptr<VBattenLearner>  learner;
    std::shared_ptr<PhysicalDataset> dataset;
    std::string                      last_metric_json;
    std::string                      last_mutation_json;
};

static thread_local std::string g_last_error;

static VBXHandle* ToHandle(void* h) { return static_cast<VBXHandle*>(h); }

extern "C" {

// ── creation / destruction ────────────────────────────────────────────────

void* vbx_learner_create(const char* params_json) {
    if (!params_json) { g_last_error = "params_json is null"; return nullptr; }
    try {
        auto j  = JsonParse(std::string(params_json));
        auto* h = new VBXHandle();
        h->learner = MakeLearner(JsonToParameter(j));
        return h;
    } catch (std::exception& e) { g_last_error = e.what(); return nullptr; }
}

void vbx_destroy(void* handle) {
    delete ToHandle(handle);
}

// ── data ──────────────────────────────────────────────────────────────────

int vbx_set_data(void* handle, const float* X, const float* y,
                 long nrows, long ncols) {
    if (!handle || !X || !y) { g_last_error = "null pointer"; return -1; }
    try {
        auto* h = ToHandle(handle);
        std::vector<vbx_float> data(X, X + nrows * ncols);
        std::vector<vbx_float> labels(y, y + nrows);
        FeatureMap fm;
        for (long c = 0; c < ncols; ++c) fm.Add("f" + std::to_string(c));
        h->dataset = MakeDenseDataset(std::move(data), nrows, ncols,
                                      std::move(fm), {}, std::move(labels));
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

// ── physics ───────────────────────────────────────────────────────────────

int vbx_set_physics(void* handle, const char* spec_json) {
    if (!handle || !spec_json) { g_last_error = "null pointer"; return -1; }
    try {
        auto j    = JsonParse(std::string(spec_json));
        auto spec = PhysicsSpec::FromJson(j);
        SetPhysicsOnLearner(ToHandle(handle)->learner.get(), spec);
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

// ── training ──────────────────────────────────────────────────────────────

int vbx_train(void* handle, int n_iters) {
    if (!handle) { g_last_error = "null handle"; return -1; }
    try {
        auto* h = ToHandle(handle);
        if (!h->dataset) { g_last_error = "no dataset set"; return -1; }
        h->learner->Train(*h->dataset, n_iters, nullptr);
        return 0;
    } catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

// ── prediction ────────────────────────────────────────────────────────────

int vbx_predict(void* handle, const float* X, long nrows, long ncols,
                float* out) {
    if (!handle || !X || !out) { g_last_error = "null pointer"; return -1; }
    try {
        auto* h = ToHandle(handle);
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

// ── serialization ─────────────────────────────────────────────────────────

int vbx_save(void* handle, const char* path) {
    if (!handle || !path) { g_last_error = "null pointer"; return -1; }
    try { ToHandle(handle)->learner->Save(path); return 0; }
    catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

int vbx_load(void* handle, const char* path) {
    if (!handle || !path) { g_last_error = "null pointer"; return -1; }
    try { ToHandle(handle)->learner->Load(path); return 0; }
    catch (std::exception& e) { g_last_error = e.what(); return -1; }
}

// ── introspection ─────────────────────────────────────────────────────────

float       vbx_train_loss(void* h)  { return h ? ToHandle(h)->learner->TrainLoss()  : 0.0f; }
int         vbx_num_stages(void* h)  { return h ? ToHandle(h)->learner->NumStages()  : 0; }

double vbx_get_metric(void* handle, const char* name) {
    if (!handle || !name) return -1.0;
    std::string n(name);
    auto* h = ToHandle(handle);
    if (n == "train_loss")  return h->learner->TrainLoss();
    if (n == "num_stages")  return h->learner->NumStages();
    return -1.0;
}

const char* vbx_get_mutation_log(void* handle, int stage) {
    if (!handle) return "{}";
    auto* h = ToHandle(handle);
    h->last_mutation_json = "{\"stage\":" + std::to_string(stage)
                          + ",\"num_stages\":" + std::to_string(h->learner->NumStages()) + "}";
    return h->last_mutation_json.c_str();
}

int  vbx_abi_version() { return VBATTENX_ABI_VERSION; }
const char* vbx_version_string() { return VBATTENX_VERSION_STRING; }
const char* vbx_last_error()     { return g_last_error.c_str(); }

} // extern "C"
