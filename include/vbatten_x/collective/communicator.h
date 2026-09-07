#pragma once

#include "vbatten_x/base.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace vbx {

enum class ReduceOp { Sum, Mean, Max, Min };

class Communicator {
public:
    virtual ~Communicator() = default;

    virtual int  Rank()       const = 0;
    virtual int  WorldSize()  const = 0;
    virtual bool IsRoot()     const { return Rank() == 0; }

    virtual void Allreduce(std::vector<vbx_float>& data, ReduceOp op) = 0;
    virtual void Broadcast(std::vector<vbx_float>& data, int root = 0) = 0;
    virtual void Barrier() = 0;

    virtual std::string Name() const = 0;
};

std::unique_ptr<Communicator> MakeLocalCommunicator();

} // namespace vbx
