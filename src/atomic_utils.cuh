#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// On GPUs with compute capability < 6.0 (like your Quadro M2000M, sm_50),
// there is no native atomicAdd(double*, double). We emulate it with atomicCAS.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ __forceinline__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(address);

    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = old_val + val;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
