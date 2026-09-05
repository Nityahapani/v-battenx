#include "vbatten_x/learner.h"
#include "vbatten_x/data.h"
#include "vbatten_x/objective.h"
#include "vbatten_x/metric.h"
#include "vbatten_x/physics_evaluator.h"
#include "vbatten_x/field_booster.h"
#include "vbatten_x/predictor.h"
#include "vbatten_x/json_io.h"
#include "src/field/field_state.h"
#include "src/booster/variational/ensemble.cc"
#include "src/booster/variational/shrinkage.cc"
#include "src/encoder/encoder_param.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace vbx {

extern std::unique_ptr<Objective>        MakeRegressionObjective();
extern std::unique_ptr<Objective>        MakeClassificationObjective();
extern std::unique_ptr<PhysicsMetric>    MakeRmseMetric();
extern std::unique_ptr<PhysicsMetric>    MakeAucMetric();
extern std::unique_ptr<PhysicsEvaluator> MakeNullEvaluator();
extern std::unique_ptr<FieldBooster>     MakeLinearBooster(vbx_float);
extern std::unique_ptr<FieldPredictor>   MakeCpuPredictor();

class VBattenLearnerImpl : public VBattenLearner {
public:
    explicit VBattenLearnerImpl(VBXParameter p) : params_(std::move(p)) {
        std::string obj = params_.GetOr<std::string>("objective", "regression");
        lr_     = static_cast<vbx_float>(params_.GetOr<double>("learning_rate", 0.1));
        lambda_ = static_cast<vbx_float>(params_.GetOr<double>("reg_lambda", 1.0));
        tol_    = static_cast<vbx_float>(params_.GetOr<double>("tol", 1e-5));
        verbose_ = params_.GetOr<int>("verbose", 1);

        if (obj == "classification") {
            obj_    = MakeClassificationObjective();
            metric_ = MakeAucMetric();
        } else {
            obj_    = MakeRegressionObjective();
            metric_ = MakeRmseMetric();
        }
        evaluator_ = MakeNullEvaluator();
        booster_   = MakeLinearBooster(lambda_);
        predictor_ = MakeCpuPredictor();
    }

    void Train(const PhysicalDataset& ds,
               int num_iters,
               EvalCallback cb) override {
        ensemble_.Clear();
        vbx_index nrows = ds.NumRows();
        pred_.assign(nrows, 0.0f);

        float prev_loss = 1e30f;
        for (int iter = 0; iter < num_iters; ++iter) {
            auto gp = obj_->GetGradients(
                {pred_.data(), (std::size_t)nrows},
                {ds.Labels(),  (std::size_t)nrows});

            FieldState stage = booster_->DoBoost(ds, gp);

            auto stage_pred = predictor_->Predict(ds, stage);
            for (vbx_index r = 0; r < nrows; ++r)
                pred_[r] += lr_ * stage_pred[r];

            ensemble_.Append(std::move(stage), lr_);

            train_loss_ = obj_->Loss(
                {pred_.data(), (std::size_t)nrows},
                {ds.Labels(),  (std::size_t)nrows});

            EvalResult er;
            er.metric_name = metric_->Name();
            er.value = metric_->Eval(
                {pred_.data(), (std::size_t)nrows},
                {ds.Labels(),  (std::size_t)nrows});
            er.iteration = iter;

            if (cb) cb(iter, er);

            if (verbose_ >= 2)
                std::cout << "[" << iter << "] loss=" << train_loss_
                          << " " << er.metric_name << "=" << er.value << "\n";

            if (std::abs(prev_loss - train_loss_) < tol_) break;
            prev_loss = train_loss_;
        }
    }

    std::vector<vbx_float> Predict(const PhysicalDataset& ds) const override {
        vbx_index nrows = ds.NumRows();
        std::vector<vbx_float> out(nrows, 0.0f);
        for (int s = 0; s < ensemble_.NumStages(); ++s) {
            auto sp = predictor_->Predict(ds, ensemble_.Stage(s).state);
            for (vbx_index r = 0; r < nrows; ++r)
                out[r] += ensemble_.Stage(s).weight * sp[r];
        }
        return out;
    }

    void Save(const std::string& path) const override {
        auto root = JsonValue::Object();
        root.Set("version", JsonValue(std::string("0.1.0")));
        root.Set("num_stages", JsonValue(ensemble_.NumStages()));
        root.Set("learning_rate", JsonValue((double)lr_));
        root.Set("reg_lambda", JsonValue((double)lambda_));

        auto stages_arr = JsonValue::Array();
        for (int s = 0; s < ensemble_.NumStages(); ++s) {
            auto& entry = ensemble_.Stage(s);
            auto params = entry.state.F->Params();
            auto stage_obj = JsonValue::Object();
            stage_obj.Set("weight", JsonValue((double)entry.weight));
            auto parr = JsonValue::Array();
            for (auto v : params) parr.Append(JsonValue((double)v));
            stage_obj.Set("params", std::move(parr));
            stages_arr.Append(std::move(stage_obj));
        }
        root.Set("stages", std::move(stages_arr));
        WriteFile(path, root.Dump());
    }

    void Load(const std::string& path) override {
        auto j = JsonParse(ReadFile(path));
        ensemble_.Clear();
        int ns = j["num_stages"].AsInt();
        lr_    = static_cast<vbx_float>(j["learning_rate"].AsDouble());
        auto& sarr = j["stages"];
        for (int s = 0; s < ns; ++s) {
            auto& sj = sarr[s];
            vbx_float w = static_cast<vbx_float>(sj["weight"].AsDouble());
            std::size_t np = sj["params"].ArraySize();
            std::vector<vbx_float> params(np);
            for (std::size_t i = 0; i < np; ++i)
                params[i] = static_cast<vbx_float>(sj["params"][i].AsDouble());

            FieldState fs;
            extern std::unique_ptr<LatentField> MakeContinuousField(std::size_t, std::size_t);
            fs.F = MakeContinuousField(np, 1);
            fs.F->Embed({params.data(), np});
            extern std::unique_ptr<FieldTopology> MakeRegionGraph();
            fs.K = MakeRegionGraph();
            extern std::unique_ptr<DimensionMap> MakeUniformDim(vbx_dim_t, std::size_t, float);
            fs.d = MakeUniformDim(static_cast<vbx_dim_t>(np), 1, 1024.0f);
            extern std::unique_ptr<TensorField> MakeRank2Tensor(vbx_region_id, vbx_dim_t, vbx_dim_t);
            fs.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(np), static_cast<vbx_dim_t>(np));
            ensemble_.Append(std::move(fs), w);
        }
    }

    int   NumStages()  const override { return ensemble_.NumStages(); }
    float TrainLoss()  const override { return train_loss_; }

private:
    VBXParameter                    params_;
    std::unique_ptr<Objective>      obj_;
    std::unique_ptr<PhysicsMetric>  metric_;
    std::unique_ptr<PhysicsEvaluator> evaluator_;
    std::unique_ptr<FieldBooster>   booster_;
    std::unique_ptr<FieldPredictor> predictor_;
    Ensemble                        ensemble_;
    std::vector<vbx_float>          pred_;
    vbx_float                       lr_;
    vbx_float                       lambda_;
    vbx_float                       tol_;
    float                           train_loss_ = 0.0f;
    int                             verbose_;
};

std::unique_ptr<VBattenLearner> MakeLearner(VBXParameter params) {
    return std::make_unique<VBattenLearnerImpl>(std::move(params));
}

} // namespace vbx
