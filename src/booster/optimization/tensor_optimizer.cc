#include "vbatten_x/base.h"
#include "vbatten_x/tensor_field.h"
#include "src/booster/optimization/field_optimizer.cc"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace vbx {

class TensorOptimizer {
public:
    TensorOptimizer(float lr = 1e-3f, float nuclear_lambda = 0.0f)
        : lr_(lr), nuclear_lambda_(nuclear_lambda) {}

    void Step(TensorField& tensor, const Eigen::VectorXf& grad) {
        vbx_region_id rid = tensor.Region();
        if (optimizers_.find(rid) == optimizers_.end())
            optimizers_.emplace(rid, AdamOptimizer(lr_));

        auto& opt   = optimizers_.at(rid);
        auto  data  = tensor.Data();
        int   n     = static_cast<int>(data.size());
        Eigen::Map<Eigen::VectorXf> params(data.data(), n);
        Eigen::VectorXf g = grad.head(std::min(grad.size(), (Eigen::Index)n));

        if (nuclear_lambda_ > 0.0f) {
            // Nuclear norm subgradient: lambda * U * V^T for SVD T = USV^T
            int side = static_cast<int>(std::sqrt(static_cast<float>(n)));
            if (side * side == n) {
                Eigen::Map<const Eigen::MatrixXf> mat(params.data(), side, side);
                Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat,
                    Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::MatrixXf sub_grad = nuclear_lambda_
                    * (svd.matrixU() * svd.matrixV().transpose());
                Eigen::Map<const Eigen::VectorXf> sub_flat(sub_grad.data(), n);
                if (g.size() == n) g += sub_flat;
            }
        }

        opt.Step(params, g);
    }

    void Reset() { optimizers_.clear(); }

private:
    float                                       lr_;
    float                                       nuclear_lambda_;
    std::unordered_map<vbx_region_id, AdamOptimizer> optimizers_;
};

} // namespace vbx
