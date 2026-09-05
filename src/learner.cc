#include "vbatten_x/learner.h"
#include "vbatten_x/data.h"
#include "vbatten_x/objective.h"
#include "vbatten_x/metric.h"
#include "vbatten_x/physics_evaluator.h"
#include "vbatten_x/physics_spec.h"
#include "vbatten_x/field_booster.h"
#include "vbatten_x/predictor.h"
#include "vbatten_x/json_io.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include "src/booster/variational/ensemble.cc"
#include "src/booster/variational/shrinkage.cc"
#include "src/encoder/encoder_param.h"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace vbx {

std::unique_ptr<Objective>        MakeRegressionObjective();
std::unique_ptr<Objective>        MakeClassificationObjective();
std::unique_ptr<PhysicsMetric>    MakeRmseMetric();
std::unique_ptr<PhysicsMetric>    MakeAucMetric();
std::unique_ptr<PhysicsMetric>    MakePdeL2Metric();
std::unique_ptr<PhysicsEvaluator> MakeNullEvaluator();
std::unique_ptr<PhysicsEvaluator> MakeEvaluatorFromMeta(const PhysicalMetaInfo&);
std::unique_ptr<FieldBooster>     MakeLinearBooster(vbx_float);
std::unique_ptr<FieldPredictor>   MakeCpuPredictor();

class VBattenLearnerImpl : public VBattenLearner {
public:
    explicit VBattenLearnerImpl(VBXParameter p) : params_(std::move(p)) {
        std::string obj = params_.GetOr<std::string>("objective", "regression");
        lr_       = static_cast<vbx_float>(params_.GetOr<double>("learning_rate", 0.1));
        lambda_   = static_cast<vbx_float>(params_.GetOr<double>("reg_lambda",    1.0));
        tol_      = static_cast<vbx_float>(params_.GetOr<double>("tol",           1e-6));
        lambda_pde_ = static_cast<vbx_float>(params_.GetOr<double>("lambda_pde",  0.0));
        verbose_  = params_.GetOr<int>("verbose", 1);

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

    void SetPhysicsSpec(const PhysicsSpec& spec) {
        evaluator_ = MakeEvaluatorFromMeta(spec.Meta());
    }

    void Train(const PhysicalDataset& ds, int num_iters, EvalCallback cb) override {
        ensemble_.Clear();
        std::size_t nrows = static_cast<std::size_t>(ds.NumRows());
        pred_.assign(nrows, 0.0f);
        train_loss_ = 0.0f;
        last_pde_residual_ = 0.0f;

        float prev_loss = 1e30f;
        for (int iter = 0; iter < num_iters; ++iter) {
            auto gp = obj_->GetGradients(
                {pred_.data(), nrows}, {ds.Labels(), nrows});

            if (lambda_pde_ > 0.0f && evaluator_) {
                FieldState tmp;
                tmp.F = MakeContinuousField(static_cast<std::size_t>(ds.NumCols()), 1);
                auto ri = evaluator_->Eval(tmp, ds);
                last_pde_residual_ = ri.MeanPde();
                float pde_grad = 2.0f * lambda_pde_ * last_pde_residual_;
                for (auto& v : gp.g) v += pde_grad;
            }

            FieldState stage = booster_->DoBoost(ds, gp);

            auto sp = predictor_->Predict(ds, stage);
            for (std::size_t r = 0; r < nrows; ++r)
                pred_[r] += lr_ * sp[r];

            ensemble_.Append(std::move(stage), lr_);

            train_loss_ = obj_->Loss({pred_.data(), nrows}, {ds.Labels(), nrows});
            if (lambda_pde_ > 0.0f)
                train_loss_ += lambda_pde_ * last_pde_residual_ * last_pde_residual_;

            EvalResult er;
            er.metric_name = metric_->Name();
            er.value       = metric_->Eval({pred_.data(), nrows}, {ds.Labels(), nrows});
            er.iteration   = iter;

            if (cb) cb(iter, er);
            if (verbose_ >= 2)
                std::cout << "[" << iter << "] loss=" << train_loss_
                          << " " << er.metric_name << "=" << er.value
                          << " pde_r=" << last_pde_residual_ << "\n";

            if (std::abs(prev_loss - train_loss_) < static_cast<float>(tol_)) break;
            prev_loss = train_loss_;
        }
    }

    std::vector<vbx_float> Predict(const PhysicalDataset& ds) const override {
        std::size_t nrows = static_cast<std::size_t>(ds.NumRows());
        std::vector<vbx_float> out(nrows, 0.0f);
        for (int s = 0; s < ensemble_.NumStages(); ++s) {
            auto& entry = ensemble_.Stage(s);
            auto sp = predictor_->Predict(ds, entry.state);
            for (std::size_t r = 0; r < nrows; ++r)
                out[r] += entry.weight * sp[r];
        }
        return out;
    }

    void Save(const std::string& path) const override {
        auto root = JsonValue::MakeObject();
        root.Set("version",       JsonValue(std::string("0.2.0")));
        root.Set("num_stages",    JsonValue(ensemble_.NumStages()));
        root.Set("learning_rate", JsonValue(static_cast<double>(lr_)));
        root.Set("reg_lambda",    JsonValue(static_cast<double>(lambda_)));
        root.Set("lambda_pde",    JsonValue(static_cast<double>(lambda_pde_)));

        auto sarr = JsonValue::MakeArray();
        for (int s = 0; s < ensemble_.NumStages(); ++s) {
            auto& entry  = ensemble_.Stage(s);
            auto  params = entry.state.F->Params();
            auto  sobj   = JsonValue::MakeObject();
            sobj.Set("weight", JsonValue(static_cast<double>(entry.weight)));
            auto parr = JsonValue::MakeArray();
            for (auto v : params) parr.Append(JsonValue(static_cast<double>(v)));
            sobj.Set("params", std::move(parr));
            sarr.Append(std::move(sobj));
        }
        root.Set("stages", std::move(sarr));
        WriteFile(path, root.Dump());
    }

    void Load(const std::string& path) override {
        auto j = JsonParse(ReadFile(path));
        ensemble_.Clear();
        lr_         = static_cast<vbx_float>(j["learning_rate"].AsDouble());
        lambda_pde_ = j.Has("lambda_pde")
                      ? static_cast<vbx_float>(j["lambda_pde"].AsDouble()) : 0.0f;
        int ns = j["num_stages"].AsInt();
        for (int s = 0; s < ns; ++s) {
            auto& sj = j["stages"][static_cast<std::size_t>(s)];
            vbx_float  w  = static_cast<vbx_float>(sj["weight"].AsDouble());
            std::size_t np = sj["params"].ArraySize();
            std::vector<vbx_float> params(np);
            for (std::size_t i = 0; i < np; ++i)
                params[i] = static_cast<vbx_float>(sj["params"][i].AsDouble());
            FieldState fs;
            fs.F = MakeContinuousField(np, 1);
            fs.F->Embed({params.data(), np});
            fs.K = MakeRegionGraph();
            fs.d = MakeUniformDim(static_cast<vbx_dim_t>(np), 1, 1024.0f);
            fs.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(np),
                                      static_cast<vbx_dim_t>(np));
            ensemble_.Append(std::move(fs), w);
        }
    }

    int   NumStages()        const override { return ensemble_.NumStages(); }
    float TrainLoss()        const override { return train_loss_; }
    float LastPdeResidual()  const          { return last_pde_residual_; }

private:
    VBXParameter                      params_;
    std::unique_ptr<Objective>        obj_;
    std::unique_ptr<PhysicsMetric>    metric_;
    std::unique_ptr<PhysicsEvaluator> evaluator_;
    std::unique_ptr<FieldBooster>     booster_;
    std::unique_ptr<FieldPredictor>   predictor_;
    Ensemble                          ensemble_;
    std::vector<vbx_float>            pred_;
    vbx_float                         lr_;
    vbx_float                         lambda_;
    vbx_float                         tol_;
    vbx_float                         lambda_pde_;
    float                             train_loss_        = 0.0f;
    float                             last_pde_residual_ = 0.0f;
    int                               verbose_;
};

std::unique_ptr<VBattenLearner> MakeLearner(VBXParameter params) {
    return std::make_unique<VBattenLearnerImpl>(std::move(params));
}

} // namespace vbx

namespace vbx {
void SetPhysicsOnLearner(VBattenLearner* learner, const PhysicsSpec& spec) {
    auto* impl = static_cast<VBattenLearnerImpl*>(learner);
    impl->SetPhysicsSpec(spec);
}
}
