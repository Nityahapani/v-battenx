#include "vbatten_x/data.h"
#include <vector>
#include <stdexcept>
#include <cstring>

namespace vbx {

class DensePhysicalDataset : public PhysicalDataset {
public:
    DensePhysicalDataset(std::vector<vbx_float> data,
                         vbx_index              nrows,
                         vbx_index              ncols,
                         FeatureMap             fmap,
                         PhysicalMetaInfo       meta,
                         std::vector<vbx_float> labels = {})
        : data_(std::move(data))
        , nrows_(nrows)
        , ncols_(ncols)
        , fmap_(std::move(fmap))
        , meta_(std::move(meta))
        , labels_(std::move(labels))
    {
        row_buf_.resize(ncols_);
        col_buf_.resize(nrows_);
    }

    vbx_index NumRows() const override { return nrows_; }
    vbx_index NumCols() const override { return ncols_; }

    Span<const vbx_float> GetRow(vbx_index row) const override {
        return {data_.data() + row * ncols_, static_cast<std::size_t>(ncols_)};
    }

    Span<const vbx_float> GetCol(vbx_index col) const override {
        for (vbx_index r = 0; r < nrows_; ++r)
            col_buf_[r] = data_[r * ncols_ + col];
        return {col_buf_.data(), static_cast<std::size_t>(nrows_)};
    }

    const vbx_float* RawData()   const override { return data_.data(); }
    const vbx_float* Labels()    const override { return labels_.data(); }
    bool              HasLabels() const override { return !labels_.empty(); }

    const FeatureMap&       Features() const override { return fmap_; }
    const PhysicalMetaInfo& Meta()     const override { return meta_; }

private:
    std::vector<vbx_float>   data_;
    vbx_index                nrows_;
    vbx_index                ncols_;
    FeatureMap               fmap_;
    PhysicalMetaInfo         meta_;
    std::vector<vbx_float>   labels_;
    mutable std::vector<vbx_float> row_buf_;
    mutable std::vector<vbx_float> col_buf_;
};

std::shared_ptr<PhysicalDataset> MakeDenseDataset(
    std::vector<vbx_float> data,
    vbx_index nrows, vbx_index ncols,
    FeatureMap fmap, PhysicalMetaInfo meta,
    std::vector<vbx_float> labels)
{
    return std::make_shared<DensePhysicalDataset>(
        std::move(data), nrows, ncols,
        std::move(fmap), std::move(meta), std::move(labels));
}

} // namespace vbx
