#include "vbatten_x/collective/communicator.h"

namespace vbx {

class LocalCommunicator : public Communicator {
public:
    int  Rank()      const override { return 0; }
    int  WorldSize() const override { return 1; }

    void Allreduce(std::vector<vbx_float>& data, ReduceOp op) override {
        (void)data; (void)op;
    }

    void Broadcast(std::vector<vbx_float>& data, int root) override {
        (void)data; (void)root;
    }

    void Barrier() override {}

    std::string Name() const override { return "local"; }
};

std::unique_ptr<Communicator> MakeLocalCommunicator() {
    return std::make_unique<LocalCommunicator>();
}

} // namespace vbx
