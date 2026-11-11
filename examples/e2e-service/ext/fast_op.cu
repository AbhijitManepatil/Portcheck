// Element-wise add. Deliberately planted; not built by default.
#include <cuda_runtime.h>

__global__ void fast_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
// Launch with: fast_add<<<blocks, threads>>>(a, b, out, n);
