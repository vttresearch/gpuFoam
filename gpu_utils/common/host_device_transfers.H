#pragma once

#include "error_handling.H"


#ifdef __USING_GPU__
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
    #else //AMD
    #include "hip/hip_runtime.h"
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
#endif



#ifdef __USING_GPU__
    #ifdef __NVIDIA_BACKEND__
    template<class InputIter, class OutputIter>
    static inline void device_to_host(InputIter d_begin, InputIter d_end, OutputIter h_begin){

        auto length = std::distance(d_begin, d_end);
        using T = typename std::iterator_traits<InputIter>::value_type;
        using T2 = typename std::iterator_traits<OutputIter>::value_type;

        static_assert(std::is_same_v<T, T2>, "Mismatching types in host_to_device");

        auto bytesize = length * sizeof(T);
        gpuErrorCheck(
            cudaMemcpy(&(*h_begin), &(*d_begin), bytesize, cudaMemcpyDeviceToHost));


    }
    #else //AMD
    #include "hip/hip_runtime.h"
    template<class InputIter, class OutputIter>
    static inline void device_to_host(InputIter d_begin, InputIter d_end, OutputIter h_begin){

        auto length = std::distance(d_begin, d_end);
        using T = typename std::iterator_traits<InputIter>::value_type;
        using T2 = typename std::iterator_traits<OutputIter>::value_type;

        static_assert(std::is_same_v<T, T2>, "Mismatching types in host_to_device");

        auto bytesize = length * sizeof(T);
        gpuErrorCheck(
            hipMemcpy(&(*h_begin), &(*d_begin), bytesize, hipMemcpyDeviceToHost));

    }
    #endif
#endif
