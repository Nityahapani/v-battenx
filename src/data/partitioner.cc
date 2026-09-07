#include "vbatten_x/data.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace vbx {

struct Partition {
    vbx_index start;
    vbx_index end;
    int       worker;
};

class RowPartitioner {
public:
    RowPartitioner(vbx_index total_rows, int world_size, uint64_t seed = 42)
        : total_rows_(total_rows), world_size_(world_size), seed_(seed) {
        order_.resize(static_cast<std::size_t>(total_rows));
        std::iota(order_.begin(), order_.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(order_.begin(), order_.end(), rng);
    }

    Partition GetPartition(int worker) const {
        vbx_index chunk = total_rows_ / world_size_;
        vbx_index start = worker * chunk;
        vbx_index end   = (worker == world_size_ - 1) ? total_rows_ : start + chunk;
        return {start, end, worker};
    }

    std::vector<vbx_index> GetIndices(int worker) const {
        auto p = GetPartition(worker);
        std::vector<vbx_index> idx;
        idx.reserve(static_cast<std::size_t>(p.end - p.start));
        for (vbx_index i = p.start; i < p.end; ++i)
            idx.push_back(order_[static_cast<std::size_t>(i)]);
        return idx;
    }

    void Reshuffle(uint64_t new_seed) {
        seed_ = new_seed;
        std::mt19937 rng(seed_);
        std::shuffle(order_.begin(), order_.end(), rng);
    }

private:
    vbx_index              total_rows_;
    int                    world_size_;
    uint64_t               seed_;
    std::vector<vbx_index> order_;
};

} // namespace vbx
