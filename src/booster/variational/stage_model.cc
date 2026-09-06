#include "src/field/field_state.h"
#include "vbatten_x/mutation_result.h"
#include "vbatten_x/base.h"
#include <vector>

namespace vbx {

struct StageModel {
    FieldState   state;
    MutationLog  mutation_log;
    vbx_float    weight;
    int          stage_idx;
    float        pde_residual_before;
    float        pde_residual_after;
};

class VariationalEnsemble {
public:
    void Append(FieldState s, MutationLog log, vbx_float w,
                float pde_before, float pde_after) {
        stages_.push_back({std::move(s), std::move(log), w,
                           static_cast<int>(stages_.size()),
                           pde_before, pde_after});
    }

    int               NumStages()  const { return static_cast<int>(stages_.size()); }
    const StageModel& Stage(int i) const { return stages_.at(static_cast<std::size_t>(i)); }
    void              Clear()            { stages_.clear(); }

    int TotalMutations() const {
        int total = 0;
        for (auto& s : stages_) total += static_cast<int>(s.mutation_log.events.size());
        return total;
    }

private:
    std::vector<StageModel> stages_;
};

} // namespace vbx
