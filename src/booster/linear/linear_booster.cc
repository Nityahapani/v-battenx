#include "vbatten_x/field_booster.h"
#include "vbatten_x/objective.h"
#include "vbatten_x/data.h"
#include "src/field/field_state.h"
#include "src/field/manifold/continuous_field.cc"
#include "src/field/topology/region_graph.cc"
#include "src/field/dimension/uniform_dim.cc"
#include "src/field/tensor/contraction_engine.cc"
#include <Eigen/Dense>
#include <vector>

namespace vbx {

extern std::unique_ptr<LatentField>   MakeContinuousField(std::size_t, std::size_t);
extern std::unique_ptr<FieldTopology> MakeRegionGraph();
extern std::unique_ptr<DimensionMap>  MakeUniformDim(vbx_dim_t, std::size_t, float);
extern std::unique_ptr<TensorField>   MakeRank2Tensor(vbx_region_id, vbx_dim_t, vbx_dim_t);

class LinearBooster : public FieldBooster {
public:
    explicit LinearBooster(vbx_float reg_lambda = 1.0f) : reg_lambda_(reg_lambda) {}

    FieldState DoBoost(const PhysicalDataset& ds, const GradPair& gp) override {
        vbx_index nrows = ds.NumRows();
        vbx_index ncols = ds.NumCols();

        Eigen::MatrixXf X(nrows, ncols + 1);
        for (vbx_index r = 0; r < nrows; ++r) {
            auto row = ds.GetRow(r);
            for (vbx_index c = 0; c < ncols; ++c)
                X(r, c) = row[c];
            X(r, ncols) = 1.0f;
        }

        Eigen::Map<const Eigen::VectorXf> g(gp.g.data(), gp.g.size());
        Eigen::Map<const Eigen::VectorXf> h(gp.h.data(), gp.h.size());

        Eigen::VectorXf hw = h;
        Eigen::MatrixXf Xw = X.array().colwise() * hw.array();
        Eigen::MatrixXf A  = Xw.transpose() * X;
        A.diagonal().array() += reg_lambda_;
        Eigen::VectorXf b = -(Xw.transpose() * g);
        weights_ = A.ldlt().solve(b);

        FieldState s;
        std::size_t ldim = ncols + 1;
        s.F = MakeContinuousField(ldim, 1);
        std::vector<vbx_float> wvec(weights_.data(), weights_.data() + weights_.size());
        s.F->Embed({wvec.data(), wvec.size()});
        s.K = MakeRegionGraph();
        s.d = MakeUniformDim(static_cast<vbx_dim_t>(ldim), 1, 1024.0f);
        s.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(ldim), static_cast<vbx_dim_t>(ldim));
        s.bias.assign(nrows, 0.0f);
        return s;
    }

    FieldPrediction PredictStage(const PhysicalDataset& ds,
                                 const FieldState& stage) const override {
        vbx_index nrows = ds.NumRows();
        vbx_index ncols = ds.NumCols();

        auto params = stage.F->Params();
        Eigen::Map<const Eigen::VectorXf> w(params.data(), params.size());

        FieldPrediction fp;
        fp.values.resize(nrows);
        for (vbx_index r = 0; r < nrows; ++r) {
            auto row = ds.GetRow(r);
            Eigen::VectorXf x(ncols + 1);
            for (vbx_index c = 0; c < ncols; ++c) x[c] = row[c];
            x[ncols] = 1.0f;
            fp.values[r] = static_cast<vbx_float>(w.dot(x));
        }
        return fp;
    }

private:
    vbx_float       reg_lambda_;
    Eigen::VectorXf weights_;
};

std::unique_ptr<FieldBooster> MakeLinearBooster(vbx_float reg_lambda) {
    return std::make_unique<LinearBooster>(reg_lambda);
}

} // namespace vbx
