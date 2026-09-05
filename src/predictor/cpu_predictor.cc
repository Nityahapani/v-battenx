#include "vbatten_x/predictor.h"
#include "vbatten_x/data.h"
#include "src/field/field_state.h"

namespace vbx {

class CpuPredictor : public FieldPredictor {
public:
    std::vector<vbx_float> Predict(const PhysicalDataset& ds,
                                   const FieldState& stage) const override {
        vbx_index nrows = ds.NumRows();
        vbx_index ncols = ds.NumCols();
        auto params = stage.F->Params();
        bool has_bias = (params.size() == static_cast<std::size_t>(ncols + 1));

        std::vector<vbx_float> out(static_cast<std::size_t>(nrows), 0.0f);
        for (vbx_index r = 0; r < nrows; ++r) {
            auto row = ds.GetRow(r);
            float v = 0.0f;
            for (vbx_index c = 0; c < ncols && c < static_cast<vbx_index>(params.size()); ++c)
                v += params[static_cast<std::size_t>(c)] * row[c];
            if (has_bias) v += params[static_cast<std::size_t>(ncols)];
            out[static_cast<std::size_t>(r)] = v;
        }
        return out;
    }
};

std::unique_ptr<FieldPredictor> MakeCpuPredictor() {
    return std::make_unique<CpuPredictor>();
}

} // namespace vbx
