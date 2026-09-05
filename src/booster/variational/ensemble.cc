#include "src/field/field_state.h"
#include "vbatten_x/base.h"
#include <vector>

namespace vbx {

struct StageEntry {
    FieldState    state;
    vbx_float     weight;
    int           stage_idx;
};

class Ensemble {
public:
    void Append(FieldState s, vbx_float weight) {
        stages_.push_back({std::move(s), weight, (int)stages_.size()});
    }

    int NumStages() const { return static_cast<int>(stages_.size()); }

    const StageEntry& Stage(int i) const { return stages_.at(i); }

    void Clear() { stages_.clear(); }

private:
    std::vector<StageEntry> stages_;
};

} // namespace vbx
