#pragma once

#include "base.h"
#include "mutation_result.h"
#include "physics_evaluator.h"
#include <memory>

namespace vbx {

struct FieldState;

struct ComplexityCost {
    int   max_total_dim   = 64;
    int   max_regions     = 16;
    int   max_connections = 32;

    int   current_total_dim   = 0;
    int   current_regions     = 0;
    int   current_connections = 0;

    static ComplexityCost FromState(const FieldState& s,
                                    int max_dim = 64,
                                    int max_reg = 16,
                                    int max_con = 32);

    int   BudgetRemaining()  const { return max_total_dim - current_total_dim; }
    bool  OverBudget()       const { return current_total_dim > max_total_dim
                                         || current_regions   > max_regions
                                         || current_connections > max_connections; }

    bool  CanAfford(MutationType t) const {
        switch (t) {
            case MutationType::Expand1dTo2d:
            case MutationType::Expand2dTo3d:
            case MutationType::LocalDimChange:
                return current_total_dim + 1 <= max_total_dim;
            case MutationType::SplitRegion:
                return current_regions + 1 <= max_regions
                    && current_connections + 1 <= max_connections;
            case MutationType::AddConnection:
                return current_connections + 1 <= max_connections;
            default:
                return true;
        }
    }
};

struct MutationResult {
    FieldState   next;
    MutationLog  log;
    bool         mutated = false;
};

class TopologicalOperator {
public:
    virtual ~TopologicalOperator() = default;

    virtual MutationResult Apply(const FieldState&    current,
                                 const ResidualInfo&  residuals,
                                 const ComplexityCost& budget) const = 0;

    virtual std::string Name() const = 0;
};

} // namespace vbx
