#include "vbatten_x/learner.h"
#include "vbatten_x/data.h"
#include "vbatten_x/objective.h"
#include "vbatten_x/metric.h"
#include "vbatten_x/physics_evaluator.h"
#include "vbatten_x/physics_spec.h"
#include "vbatten_x/field_booster.h"
#include "vbatten_x/predictor.h"
#include "vbatten_x/topological_operator.h"
#include "vbatten_x/json_io.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include "src/booster/variational/stage_model.cc"
#include "src/booster/variational/shrinkage.cc"
#include "src/encoder/encoder_param.h"
#include "src/metric/topology_metric.cc"
#include "src/metric/dimension_metric.cc"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace vbx {

std::unique_ptr<Objective>          MakeRegressionObjective();
std::unique_ptr<Objective>          MakeClassificationObjective();
std::unique_ptr<Objective>          MakeHuberObjective(vbx_float delta);
std::unique_ptr<PhysicsMetric>      MakeRmseMetric();
std::unique_ptr<PhysicsMetric>      MakeAucMetric();
std::unique_ptr<PhysicsEvaluator>   MakeNullEvaluator();
std::unique_ptr<PhysicsEvaluator>   MakeEvaluatorFromMeta(const PhysicalMetaInfo&);
std::unique_ptr<FieldBooster>       MakeLinearBooster(vbx_float);
std::unique_ptr<FieldPredictor>     MakeCpuPredictor();
std::unique_ptr<TopologicalOperator> MakeOperator(const std::string&, float, float);

class VBattenLearnerImpl : public VBattenLearner {
public:
    explicit VBattenLearnerImpl(VBXParameter p) : params_(std::move(p)) {
        std::string obj    = params_.GetOr<std::string>("objective",   "regression");
        std::string dtdo   = params_.GetOr<std::string>("dtdo",        "none");
        lr_                = static_cast<vbx_float>(params_.GetOr<double>("learning_rate", 0.1));
        lambda_            = static_cast<vbx_float>(params_.GetOr<double>("reg_lambda",    1.0));
        tol_               = static_cast<vbx_float>(params_.GetOr<double>("tol",           1e-6));
        lambda_pde_        = static_cast<vbx_float>(params_.GetOr<double>("lambda_pde",    0.0));
        tau_expand_        = static_cast<vbx_float>(params_.GetOr<double>("tau_expand",    0.1));
        tau_collapse_      = static_cast<vbx_float>(params_.GetOr<double>("tau_collapse",  0.01));
        ras_alpha_         = static_cast<vbx_float>(params_.GetOr<double>("ras_alpha",     0.0));
        huber_delta_       = static_cast<vbx_float>(params_.GetOr<double>("huber_delta",   1.0));
        max_total_dim_     = params_.GetOr<int>("max_total_dim",  64);
        max_regions_       = params_.GetOr<int>("max_regions",    16);
        max_connections_   = params_.GetOr<int>("max_connections", 32);
        verbose_           = params_.GetOr<int>("verbose",          1);

        if (obj == "classification") { obj_ = MakeClassificationObjective(); metric_ = MakeAucMetric(); }
        else if (obj == "huber")     { obj_ = MakeHuberObjective(huber_delta_); metric_ = MakeRmseMetric(); }
        else                         { obj_ = MakeRegressionObjective();     metric_ = MakeRmseMetric(); }

        evaluator_ = MakeNullEvaluator();
        booster_   = MakeLinearBooster(lambda_);
        predictor_ = MakeCpuPredictor();
        dtdo_      = MakeOperator(dtdo, tau_expand_, tau_collapse_);
    }

    void SetPhysicsSpec(const PhysicsSpec& spec) {
        evaluator_ = MakeEvaluatorFromMeta(spec.Meta());
    }

    void Train(const PhysicalDataset& ds, int num_iters, EvalCallback cb) override {
        ensemble_.Clear();
        std::size_t nrows = static_cast<std::size_t>(ds.NumRows());
        pred_.assign(nrows, 0.0f);
        train_loss_        = 0.0f;
        last_pde_residual_ = 0.0f;
        total_mutations_   = 0;

        FieldState current_state;
        current_state.F = MakeContinuousField(static_cast<std::size_t>(ds.NumCols()), 1);
        current_state.K = MakeRegionGraph();
        current_state.d = MakeUniformDim(1, 1, static_cast<float>(max_total_dim_));
        current_state.T = MakeRank2Tensor(0, 1, 1);

        float prev_loss = 1e30f;
        for (int iter = 0; iter < num_iters; ++iter) {
            ResidualInfo residuals = evaluator_->Eval(current_state, ds);
            last_pde_residual_     = residuals.MeanPde();

            MutationLog mutation_log;
            float pde_before = last_pde_residual_;

            if (dtdo_) {
                auto budget = ComplexityCost::FromState(current_state,
                                                        max_total_dim_,
                                                        max_regions_,
                                                        max_connections_);
                auto result = dtdo_->Apply(current_state, residuals, budget);
                if (result.mutated) {
                    current_state    = std::move(result.next);
                    mutation_log     = std::move(result.log);
                    total_mutations_ += static_cast<int>(mutation_log.events.size());
                }
            }

            auto gp = obj_->GetGradients({pred_.data(), nrows}, {ds.Labels(), nrows});
            if (lambda_pde_ > 0.0f) {
                float pde_grad = 2.0f * lambda_pde_ * last_pde_residual_;
                for (auto& v : gp.g) v += pde_grad;
            }

            FieldState stage = booster_->DoBoost(ds, gp);
            auto sp = predictor_->Predict(ds, stage);
            for (std::size_t r = 0; r < nrows; ++r)
                pred_[r] += lr_ * sp[r];

            float pde_after = evaluator_->Eval(current_state, ds).MeanPde();
            ensemble_.Append(std::move(stage), std::move(mutation_log),
                             lr_, pde_before, pde_after);

            train_loss_ = obj_->Loss({pred_.data(), nrows}, {ds.Labels(), nrows});
            if (lambda_pde_ > 0.0f)
                train_loss_ += lambda_pde_ * last_pde_residual_ * last_pde_residual_;

            EvalResult er;
            er.metric_name = metric_->Name();
            er.value       = metric_->Eval({pred_.data(), nrows}, {ds.Labels(), nrows});
            er.iteration   = iter;
            if (cb) cb(iter, er);

            if (verbose_ >= 2) {
                auto dm = ComputeDimMetrics(*current_state.d,
                                            current_state.K->NumRegions());
                std::cout << "[" << iter << "] loss=" << train_loss_
                          << " pde=" << last_pde_residual_
                          << " avg_dim=" << dm.avg_local_dim
                          << " dim_budget=" << dm.budget_usage_pct << "%"
                          << " mutations=" << total_mutations_ << "\n";
            }

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
        root.Set("version",          JsonValue(std::string("5.0.0")));
        root.Set("num_stages",       JsonValue(ensemble_.NumStages()));
        root.Set("learning_rate",    JsonValue(static_cast<double>(lr_)));
        root.Set("reg_lambda",       JsonValue(static_cast<double>(lambda_)));
        root.Set("lambda_pde",       JsonValue(static_cast<double>(lambda_pde_)));
        root.Set("total_mutations",  JsonValue(total_mutations_));

        auto sarr = JsonValue::MakeArray();
        for (int s = 0; s < ensemble_.NumStages(); ++s) {
            auto& entry  = ensemble_.Stage(s);
            auto  fparams = entry.state.F->Params();
            auto  sobj   = JsonValue::MakeObject();
            sobj.Set("weight",      JsonValue(static_cast<double>(entry.weight)));
            sobj.Set("pde_before",  JsonValue(static_cast<double>(entry.pde_residual_before)));
            sobj.Set("pde_after",   JsonValue(static_cast<double>(entry.pde_residual_after)));

            auto parr = JsonValue::MakeArray();
            for (auto v : fparams) parr.Append(JsonValue(static_cast<double>(v)));
            sobj.Set("params", std::move(parr));

            auto mlog = JsonValue::MakeArray();
            for (auto& ev : entry.mutation_log.events) {
                auto mobj = JsonValue::MakeObject();
                mobj.Set("type",   JsonValue(std::string(MutationTypeName(ev.type))));
                mobj.Set("region", JsonValue(static_cast<int>(ev.region_a)));
                mobj.Set("pde_r",  JsonValue(static_cast<double>(ev.residual_before)));
                mlog.Append(std::move(mobj));
            }
            sobj.Set("mutations", std::move(mlog));
            sarr.Append(std::move(sobj));
        }
        root.Set("stages", std::move(sarr));
        WriteFile(path, root.Dump());
    }

    void Load(const std::string& path) override {
        auto j = JsonParse(ReadFile(path));
        ensemble_.Clear();
        lr_              = static_cast<vbx_float>(j["learning_rate"].AsDouble());
        lambda_pde_      = j.Has("lambda_pde")
                           ? static_cast<vbx_float>(j["lambda_pde"].AsDouble()) : 0.0f;
        total_mutations_ = j.Has("total_mutations") ? j["total_mutations"].AsInt() : 0;

        int ns = j["num_stages"].AsInt();
        for (int s = 0; s < ns; ++s) {
            auto& sj  = j["stages"][static_cast<std::size_t>(s)];
            vbx_float  w  = static_cast<vbx_float>(sj["weight"].AsDouble());
            float pb  = j.Has("pde_before") ? static_cast<float>(sj["pde_before"].AsDouble()) : 0.0f;
            float pa  = j.Has("pde_after")  ? static_cast<float>(sj["pde_after"].AsDouble())  : 0.0f;
            std::size_t np = sj["params"].ArraySize();
            std::vector<vbx_float> fparams(np);
            for (std::size_t i = 0; i < np; ++i)
                fparams[i] = static_cast<vbx_float>(sj["params"][i].AsDouble());

            FieldState fs;
            fs.F = MakeContinuousField(np, 1);
            fs.F->Embed({fparams.data(), np});
            fs.K = MakeRegionGraph();
            fs.d = MakeUniformDim(static_cast<vbx_dim_t>(np), 1,
                                   static_cast<float>(max_total_dim_));
            fs.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(np),
                                      static_cast<vbx_dim_t>(np));
            ensemble_.Append(std::move(fs), MutationLog{}, w, pb, pa);
        }
    }

    int   NumStages()       const override { return ensemble_.NumStages(); }
    float TrainLoss()       const override { return train_loss_; }
    int   TotalMutations()  const          { return total_mutations_; }
    float LastPdeResidual() const          { return last_pde_residual_; }

private:
    VBXParameter                          params_;
    std::unique_ptr<Objective>            obj_;
    std::unique_ptr<PhysicsMetric>        metric_;
    std::unique_ptr<PhysicsEvaluator>     evaluator_;
    std::unique_ptr<FieldBooster>         booster_;
    std::unique_ptr<FieldPredictor>       predictor_;
    std::unique_ptr<TopologicalOperator>  dtdo_;
    VariationalEnsemble                   ensemble_;
    std::vector<vbx_float>                pred_;
    vbx_float                             lr_;
    vbx_float                             lambda_;
    vbx_float                             tol_;
    vbx_float                             lambda_pde_;
    vbx_float                             ras_alpha_;
    vbx_float                             huber_delta_;
    ResidualAdaptiveShrinkage             ras_{0.1f, 0.0f};
    float                                 tau_expand_;
    float                                 tau_collapse_;
    int                                   max_total_dim_;
    int                                   max_regions_;
    int                                   max_connections_;
    float                                 train_loss_        = 0.0f;
    float                                 last_pde_residual_ = 0.0f;
    int                                   total_mutations_   = 0;
    int                                   verbose_;
};

std::unique_ptr<VBattenLearner> MakeLearner(VBXParameter params) {
    return std::make_unique<VBattenLearnerImpl>(std::move(params));
}

void SetPhysicsOnLearner(VBattenLearner* learner, const PhysicsSpec& spec) {
    static_cast<VBattenLearnerImpl*>(learner)->SetPhysicsSpec(spec);
}

} // namespace vbx
