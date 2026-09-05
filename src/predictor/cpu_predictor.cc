#include "vbatten_x/predictor.h"
#include "vbatten_x/data.h"
#include "src/field/field_state.h"
#include <stdexcept>

namespace vbx {

class CpuPredictor : public FieldPredictor {
public:
    std::vector<vbx_float> Predict(const PhysicalDataset& ds,
                                   const FieldState& stage) const override {
        vbx_index nrows = ds.NumRows();
        vbx_index ncols = ds.NumCols();
        auto params = stage.F->Params();

        std::size_t w_size = params.size();
        std::vector<vbx_float> out(nrows, 0.0f);

        bool has_bias_col = (w_size == (std::size_t)(ncols + 1));
        for (vbx_index r = 0; r < nrows; ++r) {
            auto row = ds.GetRow(r);
            float v = 0.0f;
            for (vbx_index c = 0; c < ncols && c < (vbx_index)w_size; ++c)
                v += params[c] * row[c];
            if (has_bias_col) v += params[ncols];
            out[r] = v;
        }
        return out;
    }
};

std::unique_ptr<FieldPredictor> MakeCpuPredictor() {
    return std::make_unique<CpuPredictor>();
}

} // namespace vbx
