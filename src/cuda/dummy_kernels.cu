#include <cuda_runtime.h>

extern "C" __global__ void nano2_noop() {}

extern "C" void nano2_cuda_selftest(void) {
    nano2_noop<<<1,1>>>();
    cudaDeviceSynchronize();
}
