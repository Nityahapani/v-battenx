#ifdef VBATTENX_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "vbatten_x/base.h"
#include <stdexcept>

namespace vbx {

__global__ void rank3_contract_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

void ContractAllRegions(const float* F_dev, const float* T_dev, float* out_dev,
                         int num_regions, int dim, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dim + 15) / 16, (dim + 15) / 16);
    for (int r = 0; r < num_regions; ++r) {
        const float* Tr = T_dev + r * dim * dim;
        const float* Fr = F_dev + r * dim;
        float*       Or = out_dev + r * dim;
        rank3_contract_kernel<<<grid, block, 0, stream>>>(Tr, Fr, Or, dim, 1, dim);
    }
}

} // namespace vbx
#endif // VBATTENX_CUDA
