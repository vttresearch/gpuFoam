#pragma once

#include <thrust/device_vector.h>

namespace detail{

template<class InputIt, class OutputIt>
__global__
void async_evaluate_impl(int size, InputIt i_begin, OutputIt o_begin)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size){
        o_begin[i] = i_begin[i];
    }
}


template<class InputIt, class OutputIt>
std::pair<int, int> optimalGridAndBlockSize_evaluate(size_t size, InputIt i_begin, OutputIt o_begin){

    int minGridSize = 0;
    int blockSize = 0;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, async_evaluate_impl<InputIt, OutputIt>, 0, size);

    auto gridSize = (size + blockSize - 1) / blockSize;
    //auto gridSize = minGridSize;

    return std::make_pair(gridSize,blockSize);

}


}


template<class InputIt, class OutputIt>
void async_evaluate
(
    cudaStream_t stream,
    InputIt i_begin,
    InputIt i_end,
    OutputIt o_begin
){


    auto size = std::distance(i_begin, i_end);

    auto [gridSize, blockSize] = detail::optimalGridAndBlockSize_evaluate
    (
        size, 
        i_begin,
        o_begin
    );

    detail::async_evaluate_impl<<<gridSize,blockSize,0,stream>>>
    (
        size,
        i_begin,
        o_begin
    );

}


