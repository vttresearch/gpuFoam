#pragma once

#include "error_handling.H"

#ifdef __NVIDIA_BACKEND__
template<class InputIter, class OutputIter>
static inline void host_to_device(InputIter h_begin, InputIter h_end, OutputIter d_begin){

    auto length = std::distance(h_begin, h_end);
    using T = typename std::iterator_traits<InputIter>::value_type;
    using T2 = typename std::iterator_traits<OutputIter>::value_type;

    static_assert(std::is_same_v<T, T2>, "Mismatching types in host_to_device");

    auto bytesize = length * sizeof(T);
    gpuErrorCheck(
        cudaMemcpy(d_begin, &(*h_begin), bytesize, cudaMemcpyHostToDevice));


}
#elifdef __AMD_BACKEND__
template<class InputIter, class OutputIter>
static inline void host_to_device(InputIter h_begin, InputIter h_end, OutputIter d_begin){

    auto length = std::distance(h_begin, h_end);
    using T = typename std::iterator_traits<InputIter>::value_type;
    using T2 = typename std::iterator_traits<OutputIter>::value_type;

    static_assert(std::is_same_v<T, T2>, "Mismatching types in host_to_device");

    auto bytesize = length * sizeof(T);
    gpuErrorCheck(
        hipMemcpy(d_begin, &(*h_begin), bytesize, hipMemcpyHostToDevice));

}
#endif
