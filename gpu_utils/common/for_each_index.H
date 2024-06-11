#pragma once

#include "gpu_constants.H"
#include "error_handling.H"


template<class UnaryOperation>
__global__ void for_each_index_kernel(gLabel n, UnaryOperation op)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) op(i);
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



    #ifdef __USING_GPU__
        #ifdef __NVIDIA_BACKEND__
            for_each_index_kernel<<<NBLOCKS, NTHREADS>>>(n, op);
            gpuErrorCheck(cudaGetLastError());
            gpuErrorCheck(cudaDeviceSynchronize());
        #else
            hipLaunchKernelGGL(for_each_index_kernel, dim3(NBLOCKS),
                        dim3(NTHREADS), 0, 0, n, op);

            //gpuErrorCheck(hipGetLastError());
            //gpuErrorCheck(hipDeviceSynchronize());
        #endif
    #endif
    //Should probably throw here but this file wont anyways compile without hip or cuda compiler


}