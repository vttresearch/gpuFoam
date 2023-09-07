#pragma once

#include "gpu_fields.H"

namespace detail{

template<class It1, class It2>
__global__
void async_pluseq_impl(int size, It1 lhs, It2 rhs)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < size){
        atomicAdd(&lhs[i], double(rhs[i]));
    }
}

template<class It1, class It2>
std::pair<int, int> optimalGridAndBlockSize(size_t size, It1 lhs, It2 rhs){

    int minGridSize = 0;
    int blockSize = 0;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, async_pluseq_impl<It1, It2>, 0, size);

    auto gridSize = (size + blockSize - 1) / blockSize;
    //auto gridSize = minGridSize;

    return std::make_pair(gridSize,blockSize);

}




}

template<class Rhs>
void async_pluseq
(
    cudaStream_t stream,
    gpuScalarField& lhs,
    const Rhs& rhs
)
{

    auto size = lhs.size();
    auto [gridSize, blockSize] = detail::optimalGridAndBlockSize(size, thrust::raw_pointer_cast(lhs.data()), rhs.begin());

    detail::async_pluseq_impl<<<gridSize, blockSize, 0, stream>>>
    (
        size,
        thrust::raw_pointer_cast(lhs.data()),
        rhs.begin()
    );

}
