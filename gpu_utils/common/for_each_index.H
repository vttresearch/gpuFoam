#pragma once

#include "gpu_constants.H"
#include "error_handling.H"

namespace detail{



template<class UnaryOperation>
__global__ void cuda_backend(gLabel n, UnaryOperation op) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { op(i); }
}

}

///
///@brief Evaluates op(i) for all i in range [0, n[ in parallel.
///
///@param op A unary opeartion taking a gLabel index as a parameter. 
///@param n The maximum i index (non-inclusive).
///
template<class UnaryOperation>
static inline void for_each_index(UnaryOperation op, gLabel n){

    gLabel NTHREADS = 32;
    gLabel NBLOCKS  = (n + NTHREADS - 1) / NTHREADS;
    detail::cuda_backend<<<NBLOCKS, NTHREADS>>>(n, op);

    gpuErrorCheck(cudaGetLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

}