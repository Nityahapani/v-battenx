#ifdef VBATTENX_CUDA
#include <cuda_runtime.h>
#include "vbatten_x/base.h"

namespace vbx {

__global__ void adam_kernel(float* params, float* m, float* v,
                              const float* grad, int n,
                              float lr, float beta1, float beta2,
                              float eps, float bc1, float bc2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g   = grad[idx];
    m[idx]    = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx]    = beta2 * v[idx] + (1.0f - beta2) * g * g;
    float m_h = m[idx] / bc1;
    float v_h = v[idx] / bc2;
    params[idx] -= lr * m_h / (sqrtf(v_h) + eps);
}

void GpuAdamStep(float* params_dev, float* m_dev, float* v_dev,
                  const float* grad_dev, int n,
                  float lr, float beta1, float beta2, float eps,
                  int t, cudaStream_t stream) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(t));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(t));
    int block = 256;
    int grid  = (n + block - 1) / block;
    adam_kernel<<<grid, block, 0, stream>>>(
        params_dev, m_dev, v_dev, grad_dev,
        n, lr, beta1, beta2, eps, bc1, bc2);
}

} // namespace vbx
#endif // VBATTENX_CUDA
