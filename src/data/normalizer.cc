#include "vbatten_x/data.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vbx {

struct NormStats {
    vbx_float mean;
    vbx_float stddev;
};

class Normalizer {
public:
    void Fit(const PhysicalDataset& ds) {
        vbx_index ncols = ds.NumCols();
        vbx_index nrows = ds.NumRows();
        stats_.resize(ncols);

        for (vbx_index c = 0; c < ncols; ++c) {
            auto col = ds.GetCol(c);
            double sum = 0.0, sq = 0.0;
            for (auto v : col) { sum += v; sq += (double)v * v; }
            double mean = sum / nrows;
            double var  = sq / nrows - mean * mean;
            stats_[c].mean   = static_cast<vbx_float>(mean);
            stats_[c].stddev = static_cast<vbx_float>(std::sqrt(var < 1e-12 ? 1.0 : var));
        }
    }

    void Transform(vbx_float* data, vbx_index nrows, vbx_index ncols) const {
        for (vbx_index r = 0; r < nrows; ++r)
            for (vbx_index c = 0; c < ncols; ++c)
                data[r * ncols + c] = (data[r * ncols + c] - stats_[c].mean) / stats_[c].stddev;
    }

    void InverseTransform(vbx_float* data, vbx_index nrows, vbx_index ncols) const {
        for (vbx_index r = 0; r < nrows; ++r)
            for (vbx_index c = 0; c < ncols; ++c)
                data[r * ncols + c] = data[r * ncols + c] * stats_[c].stddev + stats_[c].mean;
    }

    const std::vector<NormStats>& Stats() const { return stats_; }

private:
    std::vector<NormStats> stats_;
};

} // namespace vbx
