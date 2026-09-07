#ifdef VBATTENX_CUDA
#include <cuda_runtime.h>
#include "vbatten_x/base.h"

namespace vbx {

__global__ void laplacian_kernel(const float* u, float* out,
                                  int nx, int ny, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    float c  = u[j * nx + i];
    float xm = u[j * nx + i - 1];
    float xp = u[j * nx + i + 1];
    float ym = u[(j-1) * nx + i];
    float yp = u[(j+1) * nx + i];
    out[j * nx + i] = (xm + xp + ym + yp - 4.0f * c) / (h * h);
}

__global__ void heat_residual_kernel(const float* u_curr, const float* u_prev,
                                      float* residual,
                                      int nx, int ny, float h,
                                      float alpha, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1) return;
    int idx = j * nx + i;
    float c  = u_curr[idx];
    float xm = u_curr[j * nx + i - 1];
    float xp = u_curr[j * nx + i + 1];
    float ym = u_curr[(j-1) * nx + i];
    float yp = u_curr[(j+1) * nx + i];
    float lap   = (xm + xp + ym + yp - 4.0f * c) / (h * h);
    float dudt  = (c - u_prev[idx]) / dt;
    residual[idx] = dudt - alpha * lap;
}

void GpuLaplacian(const float* u, float* out,
                   int nx, int ny, float h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);
    laplacian_kernel<<<grid, block, 0, stream>>>(u, out, nx, ny, h);
}

void GpuHeatResidual(const float* u_curr, const float* u_prev, float* residual,
                      int nx, int ny, float h, float alpha, float dt,
                      cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);
    heat_residual_kernel<<<grid, block, 0, stream>>>(
        u_curr, u_prev, residual, nx, ny, h, alpha, dt);
}

} // namespace vbx
#endif // VBATTENX_CUDA
