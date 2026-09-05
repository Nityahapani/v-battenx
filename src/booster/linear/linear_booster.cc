#include "vbatten_x/field_booster.h"
#include "vbatten_x/data.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <vector>

namespace vbx {

class LinearBooster : public FieldBooster {
public:
    explicit LinearBooster(vbx_float reg_lambda = 1.0f) : reg_lambda_(reg_lambda) {}

    FieldState DoBoost(const PhysicalDataset& ds, const GradPair& gp) override {
        vbx_index nrows = ds.NumRows();
        vbx_index ncols = ds.NumCols();
        int N = static_cast<int>(nrows);
        int C = static_cast<int>(ncols);

        Eigen::MatrixXf X(N, C + 1);
        for (int r = 0; r < N; ++r) {
            auto row = ds.GetRow(r);
            for (int c = 0; c < C; ++c) X(r, c) = row[c];
            X(r, C) = 1.0f;
        }

        Eigen::Map<const Eigen::VectorXf> g(gp.g.data(), N);
        Eigen::Map<const Eigen::VectorXf> h(gp.h.data(), N);

        Eigen::MatrixXf Xw = X.array().colwise() * h.array();
        Eigen::MatrixXf A  = Xw.transpose() * X;
        A.diagonal().array() += reg_lambda_;
        Eigen::VectorXf b_vec = -(Xw.transpose() * g);
        weights_ = A.ldlt().solve(b_vec);

        std::size_t ldim = static_cast<std::size_t>(C + 1);
        FieldState s;
        s.F = MakeContinuousField(ldim, 1);
        std::vector<vbx_float> wvec(weights_.data(), weights_.data() + weights_.size());
        s.F->Embed({wvec.data(), wvec.size()});
        s.K = MakeRegionGraph();
        s.d = MakeUniformDim(static_cast<vbx_dim_t>(ldim), 1, 1024.0f);
        s.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(ldim), static_cast<vbx_dim_t>(ldim));
        s.bias.assign(static_cast<std::size_t>(nrows), 0.0f);
        return s;
    }

    FieldPrediction PredictStage(const PhysicalDataset& ds,
                                 const FieldState& stage) const override {
        vbx_index nrows = ds.NumRows();
        vbx_index ncols = ds.NumCols();
        auto params = stage.F->Params();
        Eigen::Map<const Eigen::VectorXf> w(params.data(),
                                             static_cast<int>(params.size()));

        FieldPrediction fp;
        fp.values.resize(static_cast<std::size_t>(nrows));
        for (int r = 0; r < static_cast<int>(nrows); ++r) {
            auto row = ds.GetRow(r);
            Eigen::VectorXf x(static_cast<int>(ncols) + 1);
            for (int c = 0; c < static_cast<int>(ncols); ++c) x[c] = row[c];
            x[static_cast<int>(ncols)] = 1.0f;
            fp.values[static_cast<std::size_t>(r)] = w.dot(x);
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
